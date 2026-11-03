module nautilus_ftorch_interface
    !--------------------------------------------------------------------
    ! FTorch interface for the NAUTILUS full-column Conv2D+LSTM tracer
    ! model (OceanConv2DAsymXPU), exported via export_nautilus_model.py.
    !
    ! This is NOT a drop-in replacement for the CAMP ftorch_interface.F90:
    ! the NAUTILUS model is a spatiotemporal model over a gridded region
    ! (H x W), all 75 depth levels (Z) at once, with a T-step time window
    ! and 6 input channels (tracer, u, v, w, diffusivity, MLD-distance),
    ! instead of a single flat feature vector.
    !
    ! Key differences from the CAMP interface this replaces:
    !   - No per-species ppm<->physical unit conversion (preprocess/
    !     postprocess in the CAMP version). Ocean tracer values are used
    !     in whatever physical units NAUTILUS was trained on.
    !   - Two different normalizations are used across channels:
    !       * tracer            : per-depth-level min-max, using arrays
    !                              tracer_min(Z)/tracer_max(Z)
    !       * u, v, w, dif, mld  : scalar z-score, (x - mean) / std
    !   - The model is NOT stateful across forward() calls (its internal
    !     ConvLSTM hidden state is reset to zero every call). Temporal
    !     context instead comes from feeding it the last T raw time
    !     steps at once, so THIS MODULE keeps a rolling history buffer
    !     of length T and shifts it every push_timestep() call. forward()
    !     only produces a prediction once the buffer is full (T steps of
    !     data have been pushed).
    !   - MLD-distance is a DERIVED channel (depth[z] - mlotst[y,x]), not
    !     a raw model field, and is computed here in push_timestep().
    !
    ! IMPORTANT - VERIFY BEFORE PRODUCTION USE:
    ! The Fortran<->PyTorch tensor layout convention used below mirrors
    ! the one in the original CAMP ftorch_interface.F90 (Fortran array
    ! dimensions declared in the SAME order as the PyTorch tensor, with
    ! an identity `tensor_layout`). That worked for the simple (1,
    ! n_features) case. This module extends it to a 4D flattened tensor
    ! (1, T*C*Z, H, W); the index formula used to pack (t,c,z) into the
    ! flattened axis is documented in push_timestep()/forward() and MUST
    ! match exactly the `x_flat.view(B, T, C, Z, H, W)` reshape used in
    ! export_nautilus_model.py's FTorchExportWrapper. Do not trust this
    ! blindly -- run the numeric cross-check described in the
    ! accompanying README (same input in Python vs Fortran, compare
    ! outputs) before using this for real predictions.
    !--------------------------------------------------------------------
    use, intrinsic :: iso_fortran_env, only: sp => real32, dp => real64, int64
    use ftorch, only: torch_model, torch_tensor, &
                       torch_kCPU, torch_kCUDA, &
                       torch_tensor_from_array, torch_model_load, torch_model_forward, &
                       torch_delete
#ifdef NAUTILUS_USE_JSON
    use json_module, only: json_file, json_ck, json_rk, json_ik
#endif
    implicit none

    integer, parameter :: wp = sp
    real(wp), parameter :: NAUTILUS_FILL = -999.0_wp   ! land / missing sentinel

    ! ---- Dimensions, read from metadata JSON at load_model() time ----
    integer :: T_steps      ! input_steps_T   (LSTM window length)
    integer :: n_channels   ! channels_C      (tracer + forcing vars)
    integer :: n_depths     ! depth_levels_Z
    integer :: grid_H       ! height_H
    integer :: grid_W       ! width_W

    ! ---- Channel metadata (index 1 is always the tracer channel) ----
    character(len=64), allocatable :: channel_name(:)
    character(len=32), allocatable :: channel_role(:)      ! 'tracer','u','v','w','dif','mld'
    character(len=32), allocatable :: channel_norm(:)       ! 'minmax_per_depth' | 'zscore_scalar'
    real(dp), allocatable :: tracer_min(:), tracer_max(:)   ! (n_depths), channel 1 only
    real(dp), allocatable :: channel_mean(:), channel_std(:) ! (n_channels), used for channels 2..N

    real(dp), allocatable :: depths_m(:)     ! (n_depths), physical depth of each level

    ! ---- Torch objects ----
    type(torch_model) :: model
    type(torch_tensor), dimension(:), allocatable :: in_tensors
    type(torch_tensor), dimension(:), allocatable :: out_tensors
    integer :: device_type

    ! ---- Rolling time-window history buffer: (Z, H, W, C, T) ----
    ! Slot T holds the most recent step; shifted left on every push.
    real(wp), allocatable :: history(:,:,:,:,:)
    integer :: n_pushed = 0     ! how many steps have been pushed so far

    ! ---- Flattened I/O tensors matching PyTorch shapes exactly ----
    ! in_data  ~ PyTorch (B=1, T*C*Z, H, W)
    ! out_data ~ PyTorch (B=1, Z, H, W)
    real(wp), dimension(:,:,:,:), target, allocatable :: in_data
    real(wp), dimension(:,:,:,:), target, allocatable :: out_data
    integer, allocatable :: in_layout(:), out_layout(:)

contains

    !> Load model + metadata written by export_nautilus_model.py
    subroutine load_model(metadata_path)
        character(len=*), intent(in) :: metadata_path
        character(len=:), allocatable :: model_path
        character(len=:), allocatable :: model_dir
        integer :: i, slash_pos
#ifdef NAUTILUS_USE_JSON
        type(json_file) :: j_file
        logical :: found
        character(kind=json_ck, len=:), allocatable :: strval
        real(json_rk), allocatable :: tmp_array(:)
        integer(json_ik) :: n_ch
        character(len=160) :: path

        device_type = torch_kCPU

        call j_file%initialize()
        call j_file%load(filename=trim(metadata_path))
        if (j_file%failed()) then
            call j_file%print_error_message()
            print *, "ERROR: could not read/parse metadata JSON: ", trim(metadata_path)
            stop 1
        end if

        call need_int('tensor_shapes.input_steps_T', T_steps)
        call need_int('tensor_shapes.channels_C',    n_channels)
        call need_int('tensor_shapes.depth_levels_Z', n_depths)
        call need_int('tensor_shapes.height_H',      grid_H)
        call need_int('tensor_shapes.width_W',       grid_W)

        call need_str('model_path', strval)

        ! Resolve model_path relative to the metadata file's directory.
        slash_pos = index(trim(metadata_path), '/', back=.true.)
        if (slash_pos > 0) then
            model_dir = metadata_path(1:slash_pos)
        else
            model_dir = ''
        end if
        model_path = model_dir // trim(strval)

        ! ---- channels array ----
        call j_file%info('channels', found=found, n_children=n_ch)
        if (.not. found) then
            print *, "ERROR: 'channels' missing from metadata"
            stop 1
        end if
        if (int(n_ch) /= n_channels) then
            print *, "ERROR: metadata has ", int(n_ch), " channels but tensor_shapes.channels_C = ", n_channels
            stop 1
        end if

        allocate(channel_name(n_channels))
        allocate(channel_role(n_channels))
        allocate(channel_norm(n_channels))
        allocate(channel_mean(n_channels))
        allocate(channel_std(n_channels))
        if (allocated(tracer_min)) deallocate(tracer_min)
        if (allocated(tracer_max)) deallocate(tracer_max)

        do i = 1, n_channels
            write(path, '(a,i0,a)') 'channels(', i, ').'

            call need_str(trim(path)//'name', strval)
            channel_name(i) = trim(strval)

            call need_str(trim(path)//'role', strval)
            channel_role(i) = trim(strval)

            call need_str(trim(path)//'normalization', strval)
            channel_norm(i) = trim(strval)

            if (trim(channel_norm(i)) == 'minmax_per_depth') then
                call need_realvec(trim(path)//'min', tmp_array)
                if (size(tmp_array) /= n_depths) then
                    print *, "ERROR: tracer 'min' has ", size(tmp_array), " entries, expected ", n_depths
                    stop 1
                end if
                allocate(tracer_min(n_depths))
                tracer_min = tmp_array

                call need_realvec(trim(path)//'max', tmp_array)
                if (size(tmp_array) /= n_depths) then
                    print *, "ERROR: tracer 'max' has ", size(tmp_array), " entries, expected ", n_depths
                    stop 1
                end if
                allocate(tracer_max(n_depths))
                tracer_max = tmp_array

                channel_mean(i) = 0.0_dp
                channel_std(i)  = 1.0_dp
            else
                call need_real(trim(path)//'mean', channel_mean(i))
                call need_real(trim(path)//'std',  channel_std(i))
            end if
        end do

        if (.not. allocated(tracer_min) .or. .not. allocated(tracer_max)) then
            print *, "ERROR: no channel with normalization 'minmax_per_depth' (the tracer) found in metadata."
            stop 1
        end if

        ! ---- depths_m (needed to compute MLD-distance) ----
        if (allocated(depths_m)) deallocate(depths_m)
        call j_file%get('depths_m', tmp_array, found)
        if (found .and. allocated(tmp_array)) then
            if (size(tmp_array) == n_depths) then
                allocate(depths_m(n_depths))
                depths_m = tmp_array
            end if
        end if
        if (.not. allocated(depths_m)) then
            print *, "WARNING: valid 'depths_m' not in metadata; set module variable depths_m ", &
                     "(size n_depths, metres) before calling push_timestep."
        end if

        call j_file%destroy()

        ! ---- Load the TorchScript model ----
        call torch_model_load(model, trim(model_path), device_type)
        print *, "NAUTILUS model loaded from ", trim(model_path)
        print *, "T=", T_steps, " C=", n_channels, " Z=", n_depths, &
                 " H=", grid_H, " W=", grid_W

        ! ---- Allocate history buffer and I/O tensors ----
        if (allocated(history)) deallocate(history, in_data, out_data, in_tensors, out_tensors, in_layout, out_layout)
        allocate(history(n_depths, grid_H, grid_W, n_channels, T_steps))
        history = 0.0_wp
        n_pushed = 0

        allocate(in_data(1, T_steps * n_channels * n_depths, grid_H, grid_W))
        allocate(out_data(1, n_depths, grid_H, grid_W))
        allocate(in_tensors(1))
        allocate(out_tensors(1))
        allocate(in_layout(4));  in_layout  = [1, 2, 3, 4]
        allocate(out_layout(4)); out_layout = [1, 2, 3, 4]

    contains

        subroutine need_int(key, val)
            character(len=*), intent(in) :: key
            integer, intent(out) :: val
            integer(json_ik) :: tmp
            logical :: ok
            call j_file%get(key, tmp, ok)
            if (.not. ok) call missing(key)
            val = int(tmp)
        end subroutine need_int

        subroutine need_real(key, val)
            character(len=*), intent(in) :: key
            real(dp), intent(out) :: val
            real(json_rk) :: tmp
            logical :: ok
            call j_file%get(key, tmp, ok)
            if (.not. ok) call missing(key)
            val = real(tmp, dp)
        end subroutine need_real

        subroutine need_realvec(key, vec)
            character(len=*), intent(in) :: key
            real(json_rk), allocatable, intent(out) :: vec(:)
            logical :: ok
            call j_file%get(key, vec, ok)
            if (.not. ok .or. .not. allocated(vec)) call missing(key)
        end subroutine need_realvec

        subroutine need_str(key, val)
            character(len=*), intent(in) :: key
            character(kind=json_ck, len=:), allocatable, intent(out) :: val
            logical :: ok
            call j_file%get(key, val, ok)
            if (.not. ok) call missing(key)
        end subroutine need_str

        subroutine missing(key)
            character(len=*), intent(in) :: key
            print *, "ERROR: required key '", trim(key), "' missing from metadata: ", trim(metadata_path)
            stop 1
        end subroutine missing

#else
        print *, "ERROR: JSON support not available (build with -DNAUTILUS_USE_JSON). ", &
                 "Cannot load NAUTILUS metadata without it."
        stop 1
#endif
    end subroutine load_model

    !> Push one new raw time step into the rolling history buffer.
    !> All spatial fields are (Z, H, W) except mld_raw which is (H, W)
    !> (mixed-layer depth is a 2-D surface field, broadcast over depth).
    !> land_mask is (H, W), .true. = ocean, .false. = land.
    subroutine push_timestep(tracer_raw, u_raw, v_raw, w_raw, dif_raw, mld_raw, land_mask)
        real(wp), intent(in) :: tracer_raw(:,:,:)   ! (Z,H,W)
        real(wp), intent(in) :: u_raw(:,:,:)         ! (Z,H,W)
        real(wp), intent(in) :: v_raw(:,:,:)         ! (Z,H,W)
        real(wp), intent(in) :: w_raw(:,:,:)         ! (Z,H,W)
        real(wp), intent(in) :: dif_raw(:,:,:)       ! (Z,H,W)
        real(wp), intent(in) :: mld_raw(:,:)         ! (H,W)
        logical,  intent(in) :: land_mask(:,:)       ! (H,W)

        real(wp), allocatable :: mld_distance(:,:,:) ! (Z,H,W)
        real(wp) :: land_val
        integer :: z, h, wi, c

        if (.not. allocated(depths_m)) then
            print *, "ERROR: depths_m not set; cannot compute MLD-distance. ", &
                     "Call load_model() with metadata containing 'depths_m' first."
            stop 1
        end if

        allocate(mld_distance(n_depths, grid_H, grid_W))
        do z = 1, n_depths
            do wi = 1, grid_W
                do h = 1, grid_H
                    if (land_mask(h, wi)) then
                        mld_distance(z, h, wi) = real(depths_m(z) - real(mld_raw(h, wi), dp), wp)
                    else
                        mld_distance(z, h, wi) = 0.0_wp   ! overwritten below
                    end if
                end do
            end do
        end do

        ! Shift buffer: drop oldest step (slot 1), free up slot T_steps.
        if (T_steps > 1) then
            history(:,:,:,:,1:T_steps-1) = history(:,:,:,:,2:T_steps)
        end if

        ! Insert normalized fields into the newest slot (T_steps).
        do c = 1, n_channels
            select case (trim(channel_role(c)))
            case ('tracer')
                call normalize_minmax_per_depth(tracer_raw, tracer_min, tracer_max, &
                                                 history(:,:,:,c,T_steps))
            case ('u')
                call normalize_zscore(u_raw, channel_mean(c), channel_std(c), &
                                       history(:,:,:,c,T_steps))
            case ('v')
                call normalize_zscore(v_raw, channel_mean(c), channel_std(c), &
                                       history(:,:,:,c,T_steps))
            case ('w')
                call normalize_zscore(w_raw, channel_mean(c), channel_std(c), &
                                       history(:,:,:,c,T_steps))
            case ('dif')
                call normalize_zscore(dif_raw, channel_mean(c), channel_std(c), &
                                       history(:,:,:,c,T_steps))
            case ('mld')
                call normalize_zscore(mld_distance, channel_mean(c), channel_std(c), &
                                       history(:,:,:,c,T_steps))
            case default
                print *, "WARNING: unknown channel role '", trim(channel_role(c)), &
                         "' - leaving zeros in history buffer."
            end select
        end do

        ! Land cells, matching data.py: tracer/u/v/w/dif are zeroed in RAW
        ! space before normalization, while MLD-distance is NaN on land and
        ! becomes 0 AFTER normalization (nan_to_num in __getitem__).
        do c = 1, n_channels
            do z = 1, n_depths
                select case (trim(channel_role(c)))
                case ('tracer')
                    if (tracer_max(z) - tracer_min(z) > 1.0d-12) then
                        land_val = real(-tracer_min(z) / (tracer_max(z) - tracer_min(z)), wp)
                    else
                        land_val = real(-tracer_min(z), wp)
                    end if
                case ('mld')
                    land_val = 0.0_wp
                case default
                    land_val = real(-channel_mean(c) / (channel_std(c) + 1.0d-12), wp)
                end select
                where (.not. land_mask) history(z,:,:,c,T_steps) = land_val
            end do
        end do

        deallocate(mld_distance)
        n_pushed = min(n_pushed + 1, T_steps)
    end subroutine push_timestep

    !> True once T_steps worth of real data has been pushed.
    logical function history_ready()
        history_ready = (n_pushed >= T_steps)
    end function history_ready

    subroutine normalize_minmax_per_depth(raw, vmin, vmax, out)
        real(wp), intent(in)  :: raw(:,:,:)      ! (Z,H,W)
        real(dp), intent(in)  :: vmin(:), vmax(:) ! (Z)
        real(wp), intent(out) :: out(:,:,:)      ! (Z,H,W)
        integer :: z
        real(dp) :: rng
        do z = 1, size(raw, 1)
            rng = vmax(z) - vmin(z)
            ! Constant level: data.py sets the range to 1 to avoid dividing by 0.
            if (rng <= 1.0d-12) rng = 1.0d0
            out(z,:,:) = real((real(raw(z,:,:), dp) - vmin(z)) / rng, wp)
        end do
    end subroutine normalize_minmax_per_depth

    subroutine normalize_zscore(raw, mean, std, out)
        real(wp), intent(in)  :: raw(:,:,:)
        real(dp), intent(in)  :: mean, std
        real(wp), intent(out) :: out(:,:,:)
        out = real((real(raw, dp) - mean) / (std + 1.0d-12), wp)
    end subroutine normalize_zscore

    !> Denormalize the tracer prediction back to physical units.
    subroutine denormalize_tracer(pred_norm, pred_phys)
        real(wp), intent(in)  :: pred_norm(:,:,:)  ! (Z,H,W), normalized
        real(wp), intent(out) :: pred_phys(:,:,:)  ! (Z,H,W), physical units
        integer :: z
        do z = 1, size(pred_norm, 1)
            pred_phys(z,:,:) = real(real(pred_norm(z,:,:), dp) &
                                     * (tracer_max(z) - tracer_min(z)) + tracer_min(z), wp)
        end do
    end subroutine denormalize_tracer

    !> Run one forward pass. Only valid once history_ready() is .true.
    !> pred_tracer is (Z, H, W) in physical tracer units.
    subroutine forward(pred_tracer, ok)
        real(wp), intent(out) :: pred_tracer(:,:,:)   ! (Z,H,W)
        logical,  intent(out) :: ok
        integer :: t, c, z, h, wi, flat_idx

        ok = history_ready()
        if (.not. ok) then
            print *, "NAUTILUS forward(): history buffer not yet full ", &
                     "(", n_pushed, "/", T_steps, " steps pushed); skipping."
            return
        end if

        ! Pack history(Z,H,W,C,T) into in_data(1, T*C*Z, H, W) using
        ! flat_idx = (t-1)*C*Z + (c-1)*Z + z, matching the PyTorch-side
        ! `x_flat.view(B, T, C, Z, H, W)` reshape (z fastest-varying).
        do t = 1, T_steps
            do c = 1, n_channels
                do z = 1, n_depths
                    flat_idx = (t-1) * n_channels * n_depths + (c-1) * n_depths + z
                    do wi = 1, grid_W
                        do h = 1, grid_H
                            in_data(1, flat_idx, h, wi) = history(z, h, wi, c, t)
                        end do
                    end do
                end do
            end do
        end do

        call torch_tensor_from_array(in_tensors(1), in_data, in_layout, device_type)
        call torch_tensor_from_array(out_tensors(1), out_data, out_layout, torch_kCPU)

        call torch_model_forward(model, in_tensors, out_tensors)

        ! out_data(1, Z, H, W) -> denormalize -> pred_tracer(Z,H,W)
        call denormalize_tracer(out_data(1,:,:,:), pred_tracer)
    end subroutine forward

    subroutine cleanup_model()
        call torch_delete(model)
    end subroutine cleanup_model

    !> TESTING HOOK ONLY -- not part of the normal push_timestep()/forward()
    !> flow. Runs a forward pass on a caller-supplied, already-flattened,
    !> already-normalized tensor, bypassing history/normalization entirely.
    !> Use this to numerically cross-check the Fortran<->PyTorch tensor
    !> layout against a reference produced directly from the TorchScript
    !> model in Python (see make_dummy_crosscheck_input.py /
    !> test_nautilus_crosscheck.F90). raw_in must already be shaped
    !> (1, T_steps*n_channels*n_depths, grid_H, grid_W) using the same
    !> flat_idx = (t-1)*C*Z + (c-1)*Z + z convention as forward().
    !> raw_out comes back as (1, n_depths, grid_H, grid_W), NOT
    !> denormalized -- this checks the model+layout only, not the
    !> normalization math (that's exercised separately by push_timestep).
    subroutine forward_raw(raw_in, raw_out)
        real(wp), intent(in)  :: raw_in(:,:,:,:)
        real(wp), intent(out) :: raw_out(:,:,:,:)

        in_data = raw_in
        call torch_tensor_from_array(in_tensors(1), in_data, in_layout, device_type)
        call torch_tensor_from_array(out_tensors(1), out_data, out_layout, torch_kCPU)
        call torch_model_forward(model, in_tensors, out_tensors)
        raw_out = out_data
    end subroutine forward_raw

end module nautilus_ftorch_interface

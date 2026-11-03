!> test_nautilus_realdata.F90
!>
!> Real-data test of the INTENDED usage pattern: reads RAW NEMO fields for
!> T consecutive time steps (written by make_realdata_crosscheck.py), feeds
!> them through load_model -> push_timestep x T -> forward, and writes
!>   fortran_input.bin      the normalized tensor that went into the model
!>   fortran_pred_phys.bin  the prediction in physical tracer units
!> for compare_realdata.py to check against the Python reference.
!>
!> Usage:
!>   ./test_nautilus_realdata <metadata.json> <realdata_dir>
!>
!> All files are raw float32 in Fortran element order. Shapes:
!>   raw_tracer/u/v/w/dif.bin  (Z, H, W, T)
!>   raw_mld.bin               (H, W, T)
!>   land_mask.bin             (H, W) int32, 1 = ocean, 0 = land
program test_nautilus_realdata
    use nautilus_ftorch_interface
    implicit none

    character(len=1024) :: metadata_path, data_dir
    real(wp), allocatable :: tracer(:,:,:,:), u(:,:,:,:), v(:,:,:,:), w(:,:,:,:), dif(:,:,:,:)
    real(wp), allocatable :: mld(:,:,:), pred(:,:,:)
    integer,  allocatable :: mask_i(:,:)
    logical,  allocatable :: land_mask(:,:)
    integer :: t, unit_io, ios
    logical :: ok

    if (command_argument_count() < 2) then
        print *, "Usage: test_nautilus_realdata <metadata.json> <realdata_dir>"
        stop 1
    end if
    call get_command_argument(1, metadata_path)
    call get_command_argument(2, data_dir)

    print *, "[1/4] Loading model from metadata: ", trim(metadata_path)
    call load_model(trim(metadata_path))

    allocate(tracer(n_depths, grid_H, grid_W, T_steps), u(n_depths, grid_H, grid_W, T_steps))
    allocate(v(n_depths, grid_H, grid_W, T_steps), w(n_depths, grid_H, grid_W, T_steps))
    allocate(dif(n_depths, grid_H, grid_W, T_steps), mld(grid_H, grid_W, T_steps))
    allocate(mask_i(grid_H, grid_W), land_mask(grid_H, grid_W), pred(n_depths, grid_H, grid_W))

    print *, "[2/4] Reading raw NEMO fields from: ", trim(data_dir)
    call read4('raw_tracer.bin', tracer)
    call read4('raw_u.bin', u)
    call read4('raw_v.bin', v)
    call read4('raw_w.bin', w)
    call read4('raw_dif.bin', dif)
    open(newunit=unit_io, file=trim(data_dir)//'/raw_mld.bin', access='stream', &
         form='unformatted', status='old', action='read', iostat=ios)
    if (ios /= 0) call fail('raw_mld.bin')
    read(unit_io) mld
    close(unit_io)
    open(newunit=unit_io, file=trim(data_dir)//'/land_mask.bin', access='stream', &
         form='unformatted', status='old', action='read', iostat=ios)
    if (ios /= 0) call fail('land_mask.bin')
    read(unit_io) mask_i
    close(unit_io)
    land_mask = (mask_i /= 0)
    print *, "      ocean cells:", count(land_mask), " land cells:", count(.not. land_mask)

    print *, "[3/4] Pushing ", T_steps, " real time steps"
    do t = 1, T_steps
        print *, "   step", t, ": tracer", minval(tracer(:,:,:,t)), "..", maxval(tracer(:,:,:,t)), &
                 " mld", minval(mld(:,:,t)), "..", maxval(mld(:,:,t)), " m"
        call push_timestep(tracer(:,:,:,t), u(:,:,:,t), v(:,:,:,t), w(:,:,:,t), &
                           dif(:,:,:,t), mld(:,:,t), land_mask)
        print *, "        pushed. history_ready=", history_ready()
    end do

    print *, "[4/4] Running forward()"
    call forward(pred, ok)
    if (.not. ok) then
        print *, "FAIL: forward() reported not ready."
        stop 1
    end if
    print *, "      normalized input : min=", minval(in_data), " max=", maxval(in_data)
    print *, "      prediction (phys): min=", minval(pred), " max=", maxval(pred)

    open(newunit=unit_io, file=trim(data_dir)//'/fortran_input.bin', access='stream', &
         form='unformatted', status='replace', action='write', iostat=ios)
    if (ios /= 0) call fail('fortran_input.bin')
    write(unit_io) in_data
    close(unit_io)
    open(newunit=unit_io, file=trim(data_dir)//'/fortran_pred_phys.bin', access='stream', &
         form='unformatted', status='replace', action='write', iostat=ios)
    if (ios /= 0) call fail('fortran_pred_phys.bin')
    write(unit_io) pred
    close(unit_io)

    call cleanup_model()
    print *, "Done. Wrote fortran_input.bin and fortran_pred_phys.bin to ", trim(data_dir)

contains

    subroutine read4(fname, arr)
        character(len=*), intent(in) :: fname
        real(wp), intent(out) :: arr(:,:,:,:)
        integer :: lu, stat
        open(newunit=lu, file=trim(data_dir)//'/'//fname, access='stream', &
             form='unformatted', status='old', action='read', iostat=stat)
        if (stat /= 0) call fail(fname)
        read(lu) arr
        close(lu)
        print *, "      ", fname, " min=", minval(arr), " max=", maxval(arr)
    end subroutine read4

    subroutine fail(fname)
        character(len=*), intent(in) :: fname
        print *, "ERROR: could not open ", trim(data_dir)//'/'//fname
        stop 1
    end subroutine fail

end program test_nautilus_realdata

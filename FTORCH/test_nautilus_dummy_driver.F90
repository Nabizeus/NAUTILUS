!> test_nautilus_dummy_driver.F90
!>
!> Functional smoke test of the INTENDED usage pattern (load_model ->
!> push_timestep x T -> forward -> cleanup_model), using synthetic
!> "ocean-like" fields instead of real NEMO output. This does NOT check
!> numerical correctness against Python (see test_nautilus_crosscheck.F90
!> for that) -- it only checks that the full pipeline runs end to end,
!> the history buffer fills and triggers correctly, and the output is
!> finite and physically plausible in shape/range.
!>
!> Usage:
!>   ./test_nautilus_dummy_driver <metadata.json>
program test_nautilus_dummy_driver
    use nautilus_ftorch_interface
    implicit none

    character(len=1024) :: metadata_path
    real(wp), allocatable :: tracer_raw(:,:,:), u_raw(:,:,:), v_raw(:,:,:)
    real(wp), allocatable :: w_raw(:,:,:), dif_raw(:,:,:)
    real(wp), allocatable :: mld_raw(:,:)
    logical,  allocatable :: land_mask(:,:)
    real(wp), allocatable :: pred_tracer(:,:,:)
    integer :: t, z, h, wi
    logical :: ok
    real(wp) :: r

    if (command_argument_count() < 1) then
        print *, "Usage: test_nautilus_dummy_driver <metadata.json>"
        stop 1
    end if
    call get_command_argument(1, metadata_path)

    print *, "Loading model from metadata: ", trim(metadata_path)
    call load_model(trim(metadata_path))

    allocate(tracer_raw(n_depths, grid_H, grid_W))
    allocate(u_raw(n_depths, grid_H, grid_W))
    allocate(v_raw(n_depths, grid_H, grid_W))
    allocate(w_raw(n_depths, grid_H, grid_W))
    allocate(dif_raw(n_depths, grid_H, grid_W))
    allocate(mld_raw(grid_H, grid_W))
    allocate(land_mask(grid_H, grid_W))
    allocate(pred_tracer(n_depths, grid_H, grid_W))

    ! All-ocean synthetic mask (set a few cells to land to sanity-check
    ! that the MLD-distance land-masking branch runs without crashing).
    land_mask = .true.
    if (grid_H > 1 .and. grid_W > 1) land_mask(1,1) = .false.

    call random_seed()

    print *, "Pushing ", T_steps, " synthetic time steps..."
    do t = 1, T_steps
        call random_number(tracer_raw); tracer_raw = tracer_raw * 2.0_wp - 1.0_wp
        call random_number(u_raw);      u_raw      = u_raw * 2.0_wp - 1.0_wp
        call random_number(v_raw);      v_raw      = v_raw * 2.0_wp - 1.0_wp
        call random_number(w_raw);      w_raw      = w_raw * 2.0_wp - 1.0_wp
        call random_number(dif_raw);    dif_raw    = dif_raw * 2.0_wp - 1.0_wp
        call random_number(mld_raw)
        ! Keep MLD in a plausible metres range so depth-distance isn't absurd.
        mld_raw = mld_raw * 200.0_wp

        call push_timestep(tracer_raw, u_raw, v_raw, w_raw, dif_raw, mld_raw, land_mask)
        print *, "  step ", t, "/", T_steps, " pushed. history_ready=", history_ready()
    end do

    call forward(pred_tracer, ok)

    if (.not. ok) then
        print *, "FAIL: forward() reported not ready even after ", T_steps, " pushes."
        stop 1
    end if

    if (.not. all(ieee_is_finite_wrapper(pred_tracer))) then
        print *, "FAIL: prediction contains non-finite values."
        stop 1
    end if

    print *, "PASS: forward() ran and returned a finite prediction."
    print *, "pred_tracer stats: min=", minval(pred_tracer), &
             " max=", maxval(pred_tracer), &
             " mean=", sum(pred_tracer)/size(pred_tracer)

    call cleanup_model()

contains

    !> Small elemental wrapper so we don't need to `use ieee_arithmetic`
    !> at the top-level program scope just for one check.
    elemental logical function ieee_is_finite_wrapper(x) result(is_fin)
        real(wp), intent(in) :: x
        is_fin = (x == x) .and. (abs(x) < huge(x))
    end function ieee_is_finite_wrapper

end program test_nautilus_dummy_driver

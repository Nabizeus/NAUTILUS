!> test_nautilus_benchmark.F90
!>
!> Times the model forward pass through FTorch. Loads the exported model,
!> reads the dummy input written by make_dummy_crosscheck_input.py, and calls
!> forward_raw() repeatedly on it (so normalization is not included, and the
!> Python benchmark, benchmark_python.py, can time exactly the same work).
!>
!> The number of threads libtorch uses is taken from the environment variable
!> OMP_NUM_THREADS, which the calling script sets.
!>
!> Usage:
!>   OMP_NUM_THREADS=<n> ./test_nautilus_benchmark <metadata.json> <dummy_input.bin> <reps> <warmup>
program test_nautilus_benchmark
    use, intrinsic :: iso_fortran_env, only: int64, real64
    use nautilus_ftorch_interface
    implicit none

    character(len=1024) :: metadata_path, input_path, arg
    real(wp), allocatable :: raw_in(:,:,:,:), raw_out(:,:,:,:)
    real(real64), allocatable :: ms(:)
    integer :: reps, warmup, i, unit_in, ios, stat
    integer(int64) :: c0, c1, rate
    character(len=16) :: threads

    if (command_argument_count() < 4) then
        print *, "Usage: test_nautilus_benchmark <metadata.json> <dummy_input.bin> <reps> <warmup>"
        stop 1
    end if
    call get_command_argument(1, metadata_path)
    call get_command_argument(2, input_path)
    call get_command_argument(3, arg); read(arg, *) reps
    call get_command_argument(4, arg); read(arg, *) warmup
    call get_environment_variable("OMP_NUM_THREADS", threads, status=stat)
    if (stat /= 0) threads = "default"

    call load_model(trim(metadata_path))
    allocate(raw_in(1, T_steps * n_channels * n_depths, grid_H, grid_W))
    allocate(raw_out(1, n_depths, grid_H, grid_W), ms(reps))

    open(newunit=unit_in, file=trim(input_path), access='stream', &
         form='unformatted', status='old', action='read', iostat=ios)
    if (ios /= 0) then
        print *, "ERROR: could not open input file ", trim(input_path)
        stop 1
    end if
    read(unit_in) raw_in
    close(unit_in)

    do i = 1, warmup
        call forward_raw(raw_in, raw_out)
    end do

    call system_clock(count_rate=rate)
    do i = 1, reps
        call system_clock(c0)
        call forward_raw(raw_in, raw_out)
        call system_clock(c1)
        ms(i) = 1000.0_real64 * real(c1 - c0, real64) / real(rate, real64)
        print '(a,i0,a,f10.2,a)', "   run ", i, ": ", ms(i), " ms"
    end do

    print '(a,a,a,i0,a,f10.2,a,f10.2,a,f10.2)', &
        "BENCH who=Fortran+FTorch threads=", trim(threads), " reps=", reps, &
        " mean_ms=", sum(ms) / reps, " min_ms=", minval(ms), " max_ms=", maxval(ms)

    call cleanup_model()
end program test_nautilus_benchmark

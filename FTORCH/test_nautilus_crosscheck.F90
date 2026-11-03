!> test_nautilus_crosscheck.F90
!>
!> Loads the exported NAUTILUS model via nautilus_ftorch_interface,
!> reads the dummy input written by make_dummy_crosscheck_input.py,
!> runs ONE forward pass through forward_raw() (bypassing history and
!> normalization -- this checks the model + tensor layout only), and
!> writes the result to a raw binary file for compare_outputs.py to
!> check against the Python reference output.
!>
!> Usage:
!>   ./test_nautilus_crosscheck <metadata.json> <dummy_input.bin> <fortran_output.bin>
!>
!> Binary layout convention (see make_dummy_crosscheck_input.py):
!> both files are raw float32, Fortran (column-major) element order,
!> so a plain `read(unit) array` into an array of the expected shape
!> reproduces the NumPy array exactly.
program test_nautilus_crosscheck
    use nautilus_ftorch_interface
    implicit none

    character(len=1024) :: metadata_path, input_path, output_path
    real(wp), allocatable :: raw_in(:,:,:,:)
    real(wp), allocatable :: raw_out(:,:,:,:)
    integer :: unit_in, unit_out, ios

    if (command_argument_count() < 3) then
        print *, "Usage: test_nautilus_crosscheck <metadata.json> <dummy_input.bin> <fortran_output.bin>"
        stop 1
    end if

    call get_command_argument(1, metadata_path)
    call get_command_argument(2, input_path)
    call get_command_argument(3, output_path)

    print *, "Loading model from metadata: ", trim(metadata_path)
    call load_model(trim(metadata_path))

    ! Dims (T_steps, n_channels, n_depths, grid_H, grid_W) are set by
    ! load_model() above.
    allocate(raw_in(1, T_steps * n_channels * n_depths, grid_H, grid_W))
    allocate(raw_out(1, n_depths, grid_H, grid_W))

    print *, "Reading dummy input: ", trim(input_path)
    open(newunit=unit_in, file=trim(input_path), access='stream', &
         form='unformatted', status='old', action='read', iostat=ios)
    if (ios /= 0) then
        print *, "ERROR: could not open input file ", trim(input_path)
        stop 1
    end if
    read(unit_in) raw_in
    close(unit_in)

    print *, "Input  stats: min=", minval(raw_in), " max=", maxval(raw_in), &
             " mean=", sum(raw_in)/size(raw_in)

    call forward_raw(raw_in, raw_out)

    print *, "Output stats: min=", minval(raw_out), " max=", maxval(raw_out), &
             " mean=", sum(raw_out)/size(raw_out)

    print *, "Writing Fortran output: ", trim(output_path)
    open(newunit=unit_out, file=trim(output_path), access='stream', &
         form='unformatted', status='replace', action='write', iostat=ios)
    if (ios /= 0) then
        print *, "ERROR: could not open output file ", trim(output_path)
        stop 1
    end if
    write(unit_out) raw_out
    close(unit_out)

    call cleanup_model()
    print *, "Done. Compare with: python compare_outputs.py --fortran-output ", trim(output_path)

end program test_nautilus_crosscheck

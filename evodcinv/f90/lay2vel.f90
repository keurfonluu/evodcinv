!=======================================================================
! Created by
!     Keurfon Luu <keurfon.luu@mines-paristech.fr>
!     MINES ParisTech - Centre de Géosciences
!     PSL - Research University
!=======================================================================

module lay2vel

  implicit none

contains

  subroutine lay2vel1(vel, lay, nlay, dz, nz)
    integer, intent(in) :: nlay, nz
    real, intent(in) :: lay(nlay,2), dz
    real, intent(out) :: vel(nz)
    integer :: i, ztop, zbot

    ztop = 1
    do i = 1, nlay
      zbot = min( nint( lay(i,2) / dz ), nz )
      vel(ztop:zbot) = lay(i,1)
      if ( zbot .ge. nz ) then
        exit
      else
        ztop = zbot + 1
      end if
    end do
    vel(zbot+1:nz) = lay(nlay,1)
    return
  end subroutine lay2vel1

  subroutine lay2vel2(vel, lay, nlay, dz, nz, nx)
    integer, intent(in) :: nlay, nz, nx
    real, intent(in) :: lay(nlay,2), dz
    real, intent(out) :: vel(nz, nx)
    integer :: ix
    real :: vel1d(nz)

    call lay2vel1(vel1d, lay, nlay, dz, nz)
    do ix = 1, nx
      vel(:,ix) = vel1d
    end do
    return
  end subroutine lay2vel2

  subroutine lay2vel3(vel, lay, nlay, dz, nz, nx, ny)
    integer, intent(in) :: nlay, nz, nx, ny
    real, intent(in) :: lay(nlay,2), dz
    real, intent(out) :: vel(nz, nx, ny)
    integer :: iy
    real :: vel2d(nz,nx)

    call lay2vel2(vel2d, lay, nlay, dz, nz, nx)
    do iy = 1, ny
      vel(:,:,iy) = vel2d
    end do
    return
  end subroutine lay2vel3

end module lay2vel

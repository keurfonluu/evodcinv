!=======================================================================
! Created by
!     Keurfon Luu <keurfon.luu@mines-paristech.fr>
!     MINES ParisTech - Centre de Géosciences
!     PSL - Research University
!=======================================================================

module dispcurve

  use omp_lib

  implicit none

  real(kind = 8), parameter :: pi = 3.141592653589793238460d0

contains

  function thomson_haskell(w, k, alpha, beta, rho, d, nl, wtype) result(panel)
    integer(kind = 4), intent(in) :: nl
    real(kind = 8), intent(in) :: w, k, alpha(nl), beta(nl), rho(nl), d(nl)
    character(len = *), optional :: wtype
    complex(kind = 8) :: panel
    integer(kind = 4) :: i
    real(kind = 8) :: c, X(5)
    real(kind = 8), dimension(nl) :: mu, gam, t
    complex(kind = 8) :: Tl(2,2), Tli(2,2,nl-1), U(2), V(2)
    complex(kind = 8), dimension(nl) :: Ca, Sa, Cb, Sb, r, s
    complex(kind = 8), dimension(nl-1) :: eps, eta, a, ap, b, bp
    complex(kind = 8) :: p1, p2, p3, p4
    complex(kind = 8) :: q1, q2, q3, q4
    complex(kind = 8) :: y1, y2, z1, z2

    ! Model layer parameters
    c = w / k                 ! Phase velocity
    mu = rho * beta**2        ! Layers' rigidity

    ! Layer eigenfunctions
    do i = 1, nl
      if ( trim(wtype) .eq. "rayleigh" ) then
        if ( c .lt. alpha(i) ) then
          r(i) = sqrt( 1.0d0 - c**2 / alpha(i)**2 )
          Ca(i) = cosh( k * r(i) * d(i) )
          Sa(i) = sinh( k * r(i) * d(i) )
        elseif ( c .gt. alpha(i) ) then
          r(i) = cmplx( 0.0d0, sqrt( c**2 / alpha(i)**2 - 1.0d0 ) )
          Ca(i) = cos( k * imag(r(i)) * d(i) )
          Sa(i) = cmplx( 0.0d0, sin( k * imag(r(i)) * d(i) ) )
        else
          r(i) = 0.0d0
          Ca(i) = 1.0d0
          Sa(i) = 0.0d0
        end if
      end if

      if ( c .lt. beta(i) ) then
        s(i) = sqrt( 1.0d0 - c**2 / beta(i)**2 )
        Cb(i) = cosh( k * s(i) * d(i) )
        Sb(i) = sinh( k * s(i) * d(i) )
      elseif ( c .gt. beta(i) ) then
        s(i) = cmplx( 0.0d0, sqrt( c**2 / beta(i)**2  - 1.0d0 ) )
        Cb(i) = cos( k * imag(s(i)) * d(i) )
        Sb(i) = cmplx( 0.0d0, sin( k * imag(s(i)) * d(i) ) )
      else
        s(i) = 0.0d0
        Cb(i) = 1.0d0
        Sb(i) = 0.0d0
      end if
    end do

    ! Thomson-Haskell propagator method
    select case(trim(wtype))

    ! Rayleigh-wave reduced fast Delta matrix
    case("rayleigh")
      gam = beta**2 / c**2
      t = 2.0d0 - c**2 / beta**2

      do i = 1, nl-1
        eps(i) = rho(i+1) / rho(i)
        eta(i) = 2.0d0 * ( gam(i) - eps(i) * gam(i+1) )
      end do

      a = eps + eta
      ap = a - 1.0d0
      b = 1.0d0 - eta
      bp = b - 1.0d0

      X = mu(1)*mu(1) * [ 2.0d0*t(1), -t(1)**2, 0.0d0, 0.0d0, -4.0d0 ]

      do i = 1, nl-1
        X = X
        p1 = Cb(i) * X(2) + s(i) * Sb(i) * X(3)
        p2 = Cb(i) * X(4) + s(i) * Sb(i) * X(5)
        if ( c .ne. beta(i) ) then
          p3 = 1.0d0/s(i) * Sb(i) * X(2) + Cb(i) * X(3)
          p4 = 1.0d0/s(i) * Sb(i) * X(4) + Cb(i) * X(5)
        else
          p3 = k * d(i) * X(2) + Cb(i) * X(3)
          p4 = k * d(i) * X(4) + Cb(i) * X(5)
        end if

        q1 = Ca(i) * p1 - r(i) * Sa(i) * p2
        q3 = Ca(i) * p3 - r(i) * Sa(i) * p4
        if ( c .ne. alpha(i) ) then
          q2 = -1.0d0/r(i) * Sa(i) * p3 + Ca(i) * p4
          q4 = -1.0d0/r(i) * Sa(i) * p1 + Ca(i) * p2
        else
          q2 = -k * d(i) * p3 + Ca(i) * p4
          q4 = -k * d(i) * p1 + Ca(i) * p2
        end if

        y1 = ap(i) * X(1) + a(i) * q1
        y2 = a(i) * X(1) + ap(i) * q2
        z1 = b(i) * X(1) + bp(i) * q1
        z2 = bp(i) * X(1) + b(i) * q2

        X(1) = bp(i) * y1 + b(i) * y2
        X(2) = a(i) * y1 + ap(i) * y2
        X(3) = eps(i) * q3
        X(4) = eps(i) * q4
        X(5) = bp(i) * z1 + b(i) * z2
      end do
      panel = X(2) + s(nl) * X(3) - r(nl) * ( X(4) + s(nl) * X(5) )

    ! Love-wave propagator matrix
    case("love")
      Tli(1,1,:) = Cb(:nl-1)
      Tli(1,2,:) = merge(Sb(:nl-1) / mu(:nl-1) / s(:nl-1), &
                         dcmplx(k) * dcmplx(d(:nl-1)) / mu(:nl-1), &
                         mask = c .ne. beta(:nl-1))
      Tli(2,1,:) = mu(:nl-1) * s(:nl-1) * Sb(:nl-1)
      Tli(2,2,:) = Cb(:nl-1)
      U = [ 0.d0, 1.d0 ]
      V(1) = 1.d0
      V(2) = mu(nl) * s(nl)

      Tl = Tli(:,:,1)
      do i = 2, nl-1
        Tl = Tl
        Tl = matmul(Tl, Tli(:,:,i))
      end do
      panel = dot_product(matmul(U, Tl), V)

    case default
      print *, "Error: unkown surface wave type '" // trim(wtype) // "'"
      stop
    end select
    return
  end function thomson_haskell

  function fcpanel(f, c, alpha, beta, rho, d, nf, nc, nl, wtype, n_threads) result(panel)
    integer(kind = 4), intent(in) :: nf, nc, nl
    real(kind = 8), intent(in) :: f(nf), c(nc), alpha(nl), beta(nl), rho(nl), d(nl)
    character(len = *), intent(in), optional :: wtype
    integer(kind = 4), intent(in), optional :: n_threads
    complex(kind = 8) :: panel(nc,nf)
    integer(kind = 4) :: i, j
    real(kind = 8) :: w
    character(len = 8) :: opt_wtype

    opt_wtype = "rayleigh"
    if ( present(wtype) ) opt_wtype = trim(wtype)
    if ( present(n_threads) ) call omp_set_num_threads(n_threads)

    !$omp parallel default(shared) private(w)
    !$omp do schedule(runtime)
    do j = 1, nf
      w = 2. * pi * f(j)
      do i = 1, nc
        panel(i,j) = thomson_haskell(w, w/c(i), alpha, beta, rho, d, nl, opt_wtype)
      end do
    end do
    !$omp end parallel
    return
  end function fcpanel

  function fkpanel(f, k, alpha, beta, rho, d, nf, nk, nl, wtype, n_threads) result(panel)
    integer(kind = 4), intent(in) :: nf, nk, nl
    real(kind = 8), intent(in) :: f(nf), k(nk), alpha(nl), beta(nl), rho(nl), d(nl)
    character(len = *), intent(in), optional :: wtype
    integer(kind = 4), intent(in), optional :: n_threads
    complex(kind = 8) :: panel(nk,nf)
    integer(kind = 4) :: i, j
    real(kind = 8) :: w
    character(len = 8) :: opt_wtype

    opt_wtype = "rayleigh"
    if ( present(wtype) ) opt_wtype = trim(wtype)
    if ( present(n_threads) ) call omp_set_num_threads(n_threads)

    !$omp parallel default(shared) private(w)
    !$omp do schedule(runtime)
    do j = 1, nf
      w = 2. * pi * f(j)
      do i = 1, nk
        panel(i,j) = thomson_haskell(w, k(i), alpha, beta, rho, d, nl, opt_wtype)
      end do
    end do
    !$omp end parallel
    return
  end function fkpanel

  function fkpanel_feasible(f, nk, alpha, beta, rho, d, nf, nl, wtype, n_threads) result(panel)
    integer(kind = 4), intent(in) :: nf, nk, nl
    real(kind = 8), intent(in) :: f(nf), alpha(nl), beta(nl), rho(nl), d(nl)
    character(len = *), intent(in), optional :: wtype
    integer(kind = 4), intent(in), optional :: n_threads
    complex(kind = 8) :: panel(nk,nf)
    integer(kind = 4) :: i, j, l
    real(kind = 8) :: Vmin, Vmax, kmin, kmax, dk, w, k(nk)
    character(len = 8) :: opt_wtype

    opt_wtype = "rayleigh"
    if ( present(wtype) ) opt_wtype = trim(wtype)
    if ( present(n_threads) ) call omp_set_num_threads(n_threads)

    Vmin = floor( minval( rayleigh_velocity(alpha, beta, nl) ) )
    Vmax = maxval(beta)

    !$omp parallel default(shared) private(w, kmin, kmax, dk, k)
    !$omp do schedule(runtime)
    do j = 1, nf
      w = 2. * pi * f(j)
      kmin = w / Vmax
      kmax = w / Vmin
      dk = ( kmax - kmin ) / ( nk - 1. )
      k = kmin + dk * [ ( l-1., l = 1, nk ) ]
      do i = 1, nk
        panel(i,j) = thomson_haskell(w, k(i), alpha, beta, rho, d, nl, opt_wtype)
      end do
    end do
    !$omp end parallel
    return
  end function fkpanel_feasible

  function rayleigh_velocity(alpha, beta, nl) result(v)
    integer(kind = 4), intent(in) :: nl
    real(kind = 8), intent(in) :: alpha(nl), beta(nl)
    real(kind = 8) :: v(nl), ksi(nl), nu(nl), Vr(nl)

    ksi = alpha**2 / beta**2
    nu = ( 1.0d0 - 0.5d0 * ksi ) / ( 1.0d0 - ksi )
    Vr = ( 0.87d0 + 1.12d0 * nu ) / ( 1.0d0 + nu )
    v = Vr * beta
    return
  end function rayleigh_velocity

  function pick(panel, faxis, yaxis, modes, ny, nf, nm, n_threads) result(dcurve)
    integer(kind = 4), intent(in) :: ny, nf, nm
    real(kind = 8), intent(in) :: panel(ny,nf), faxis(nf), yaxis(ny)
    integer(kind = 4), intent(in) :: modes(nm)
    integer(kind = 4), intent(in), optional :: n_threads
    integer(kind = 4) :: i, j, m
    real(kind = 8) :: dcurve(nm,2,nf)
    integer(kind = 4), dimension(:), allocatable :: idx, iy
    real(kind = 8), dimension(:), allocatable :: tmp

    if ( present(n_threads) ) call omp_set_num_threads(n_threads)

    ! Initialize variables
    dcurve = 0.d0                       ! Picked dispersion curves for each mode
    iy = [ ( i, i = 1, ny ) ]           ! Phase velocity axis (index)

    !$omp parallel default(shared) private(tmp, idx, m)
    !$omp do schedule(runtime)
    do i = 1, nf
      tmp = panel(:,i) / dabs( maxval(panel(:,i)) )
      idx = pack(iy, mask = tmp(:ny-1) * tmp(2:) .lt. 0.d0)
      do j = 1, nm
        m = modes(j) + 1
        if ( size(idx) .ge. m ) then
          dcurve(j,1,i) = ( yaxis(idx(m)) * tmp(idx(m)+1) - yaxis(idx(m)+1) * tmp(idx(m)) ) &
                          / ( tmp(idx(m)+1) - tmp(idx(m)) )
          dcurve(j,2,i) = faxis(i)
        end if
      end do
      deallocate(tmp, idx)
    end do
    !$omp end parallel
    return
  end function pick

end module dispcurve

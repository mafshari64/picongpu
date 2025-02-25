/* Copyright 2003-2024 Alexander Debus, C. Bond
 *
 * This file is part of PMacc.
 *
 * PMacc is free software: you can redistribute it and/or modify
 * it under the terms of either the GNU General Public License or
 * the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * PMacc is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License and the GNU Lesser General Public License
 * for more details.
 *
 * You should have received a copy of the GNU General Public License
 * and the GNU Lesser General Public License along with PMacc.
 * If not, see <http://www.gnu.org/licenses/>.
 */

/** @file Bessel.tpp
 *
 *  Reference: Implementation is derived from a C++ implementation of
 *             complex Bessel functions from C. Bond (2003).
 *
 *  Original source downloaded from: http://www.crbond.com
 *  Download date: 2017/07/27
 *  Files: CBESSJY.CPP, BESSEL.H
 *  File-Header:
 *      cbessjy.cpp -- complex Bessel functions.
 *      Algorithms and coefficient values from "Computation of Special
 *      Functions", Zhang and Jin, John Wiley and Sons, 1996.
 *
 *     (C) 2003, C. Bond. All rights reserved.
 *
 *  The website (http://www.crbond.com) furthermore states:
 *  "This website contains a variety of materials related to
 *  technology and engineering. Downloadable software, much of it
 *  original, is available from some of the pages. All downloadable
 *  software is offered freely and without restriction -- although
 *  in most cases the files should be considered as works in progress
 *  (alpha or beta level). Source code is also included for some
 *  applications."
 *
 *  Code history:
 *  1/03 -- Added C/C++ source files for real and complex gamma function
 *  and psi function. Also added individual C/C++ files  for Bessel and
 *  modified Bessel functions of 1st and 2nd kinds for real and complex
 *  arguments. Updated Butterworth and Bessel filter tables with files of
 *   extended parameters including polynomials, poles and component values.
 *  6/04 -- Revised bessel.zip to correct errors in complex Bessel functions.
 *
 *  Further (re-)implementation of this code has been done by FZ Juelich.
 *  URL: http://apps.jcns.fz-juelich.de/redmine/issues/569#change-2056
 *  Above URL also includes a accuracy test report of this code against
 *  SLATEC, MAPLE and MATHEMATICA.
 */

#pragma once

#include "pmacc/algorithms/math.hpp"
#include "pmacc/math/Complex.hpp"
#include "pmacc/math/Vector.hpp"
#include "pmacc/types.hpp"

#include <cmath>
#include <limits>

namespace pmacc
{
    namespace math
    {
        namespace bessel
        {
            template<typename T_Type, typename T_TableA, typename T_TableB>
            struct Cbesselj0Base
            {
                using result = alpaka::Complex<T_Type>;
                using complex_T = alpaka::Complex<T_Type>;
                using float_T = T_Type;

                HDINLINE result operator()(complex_T const& z)
                {
                    T_TableA a;
                    T_TableB b;
                    result cj0;
                    /* The target rel. accuracy goal eps is chosen according to the original implementation
                     * of C. Bond, where for double-precision the accuracy goal is 1.0e-15. Here the accuracy
                     * goal value is the same 4.5 * DBL_EPSILON = 1.0e-15 for double-precision, but is similarly
                     * defined for float-precision.
                     */
                    float_T const eps = float_T(4.5) * std::numeric_limits<float_T>::epsilon();

                    complex_T const cone = complex_T(1, 0);

                    float_T const a0 = math::abs(z);
                    complex_T const z2 = z * z;
                    complex_T z1 = z;
                    if(a0 == float_T(0.0))
                    {
                        cj0 = cone;
                        return cj0;
                    }
                    if(z.real() < float_T(0.0))
                        z1 = float_T(-1.0) * z;
                    if(a0 <= float_T(12.0))
                    {
                        cj0 = cone;
                        complex_T cr = cone;
                        for(uint32_t k = 1u; k <= 40u; k++)
                        {
                            cr *= float_T(-0.25) * z2 / float_T(k * k);
                            cj0 += cr;
                            if(math::abs(cr) < math::abs(cj0) * eps)
                                break;
                        }
                    }
                    else
                    {
                        uint32_t kz;
                        if(a0 >= float_T(50.0))
                            kz = 8u; // can be changed to 10
                        else if(a0 >= float_T(35.0))
                            kz = 10u; //   "      "     "  12
                        else
                            kz = 12u; //   "      "     "  14
                        constexpr auto quarterPi = Pi<float_T>::quarterValue;
                        complex_T ct1 = z1 - quarterPi;
                        complex_T cp0 = cone;
                        for(uint32_t k = 0u; k < kz; k++)
                        {
                            cp0 += a[k] * math::pow(z1, float_T(-2.0) * k - float_T(2.0));
                        }
                        complex_T cq0 = float_T(-0.125) / z1;
                        for(uint32_t k = 0; k < kz; k++)
                        {
                            cq0 += b[k] * math::pow(z1, float_T(-2.0) * k - float_T(3.0));
                        }
                        constexpr auto doubleReciprocalPi = Pi<float_T>::doubleReciprocalValue;
                        complex_T const cu = math::sqrt(doubleReciprocalPi / z1);
                        cj0 = cu * (cp0 * math::cos(ct1) - cq0 * math::sin(ct1));
                    }
                    return cj0;
                }
            };

            template<typename T_Type, typename T_TableA1, typename T_TableB1>
            struct Cbesselj1Base
            {
                using result = alpaka::Complex<T_Type>;
                using complex_T = alpaka::Complex<T_Type>;
                using float_T = T_Type;

                HDINLINE result operator()(complex_T const& z)
                {
                    T_TableA1 a1;
                    T_TableB1 b1;
                    result cj1;
                    /* The target rel. accuracy goal eps is chosen according to the original implementation
                     * of C. Bond, where for double-precision the accuracy goal is 1.0e-15. Here the accuracy
                     * goal value is the same 4.5 * DBL_EPSILON = 1.0e-15 for double-precision, but is similarly
                     * defined for float-precision.
                     */
                    float_T const eps = float_T(4.5) * std::numeric_limits<float_T>::epsilon();

                    complex_T const cone = complex_T(1, 0);
                    complex_T const czero = complex_T(0, 0);

                    float_T const a0 = math::abs(z);
                    complex_T const z2 = z * z;
                    complex_T z1 = z;
                    if(a0 == float_T(0.0))
                    {
                        cj1 = czero;
                        return cj1;
                    }
                    if(z.real() < float_T(0.0))
                        z1 = float_T(-1.0) * z;
                    if(a0 <= float_T(12.0))
                    {
                        cj1 = cone;
                        complex_T cr = cone;
                        for(uint32_t k = 1u; k <= 40u; k++)
                        {
                            cr *= float_T(-0.25) * z2 / (k * (k + float_T(1.0)));
                            cj1 += cr;
                            if(math::abs(cr) < math::abs(cj1) * eps)
                                break;
                        }
                        cj1 *= float_T(0.5) * z1;
                    }
                    else
                    {
                        uint32_t kz;
                        if(a0 >= float_T(50.0))
                            kz = 8u; // can be changed to 10
                        else if(a0 >= float_T(35.0))
                            kz = 10u; //   "      "     "  12
                        else
                            kz = 12u; //   "      "     "  14
                        constexpr auto doubleReciprocalPi = Pi<float_T>::doubleReciprocalValue;
                        complex_T const cu = math::sqrt(doubleReciprocalPi / z1);
                        complex_T const ct2 = z1 - float_T(0.75) * Pi<float_T>::value;
                        complex_T cp1 = cone;
                        for(uint32_t k = 0u; k < kz; k++)
                        {
                            cp1 += a1[k] * math::pow(z1, float_T(-2.0) * k - float_T(2.0));
                        }
                        complex_T cq1 = float_T(0.375) / z1;
                        for(uint32_t k = 0u; k < kz; k++)
                        {
                            cq1 += b1[k] * math::pow(z1, float_T(-2.0) * k - float_T(3.0));
                        }
                        cj1 = cu * (cp1 * math::cos(ct2) - cq1 * math::sin(ct2));
                    }
                    if(z.real() < float_T(0.0))
                    {
                        cj1 = float_T(-1.0) * cj1;
                    }
                    return cj1;
                }
            };

            template<>
            struct J0<alpaka::Complex<double>>
                : public Cbesselj0Base<double, math::bessel::aDouble_t, math::bessel::bDouble_t>
            {
            };

            template<>
            struct J0<alpaka::Complex<float>>
                : public Cbesselj0Base<float, math::bessel::aFloat_t, math::bessel::bFloat_t>
            {
            };

            template<>
            struct J1<alpaka::Complex<double>>
                : public Cbesselj1Base<double, math::bessel::a1Double_t, math::bessel::b1Double_t>
            {
            };

            template<>
            struct J1<alpaka::Complex<float>>
                : public Cbesselj1Base<float, math::bessel::a1Float_t, math::bessel::b1Float_t>
            {
            };

        } // namespace bessel
    } // namespace math
} // namespace pmacc

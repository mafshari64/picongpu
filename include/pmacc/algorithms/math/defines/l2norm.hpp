/* Copyright 2013-2024 Heiko Burau, Rene Widera
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

#pragma once

#include "pmacc/types.hpp"

namespace pmacc
{
    namespace math
    {
        /// definition must be provided by Type
        template<typename Type>
        struct L2norm;

        /** l2norm
         *
         * only defined for vectors
         *
         * @return sqrt(abs(x)^2 + ...)
         */
        template<typename T1>
        HDINLINE typename L2norm<T1>::result l2norm(const T1& value)
        {
            return L2norm<T1>()(value);
        }

        template<typename Type>
        struct L2norm2;

        /** l2norm2
         *
         * only defined for vectors
         *
         * @return abs(x)^2 + ...
         */
        template<typename T1>
        HDINLINE typename L2norm2<T1>::result l2norm2(const T1& value)
        {
            return L2norm2<T1>()(value);
        }
    } // namespace math
} // namespace pmacc

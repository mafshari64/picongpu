/* Copyright 2015-2024 Heiko Burau
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

#include "pmacc/traits/HasFlag.hpp"
#include "pmacc/types.hpp"

namespace pmacc
{
    namespace particles
    {
        namespace traits
        {
            /** Return a new sequence of particle species carrying flag.
             *
             * @tparam T_MPLSeq sequence of particle species
             * @tparam T_Flag flag to be filtered
             */
            template<typename T_MPLSeq, typename T_Flag>
            struct FilterByFlag
            {
                template<typename T_Species>
                using HasFlag = typename ::pmacc::traits::HasFlag<typename T_Species::FrameType, T_Flag>::type;

                using type = mp_copy_if<T_MPLSeq, HasFlag>;
            };

        } // namespace traits
    } // namespace particles
} // namespace pmacc

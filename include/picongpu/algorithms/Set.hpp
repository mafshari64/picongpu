/* Copyright 2013-2024 Heiko Burau, Rene Widera
 *
 * This file is part of PIConGPU.
 *
 * PIConGPU is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * PIConGPU is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with PIConGPU.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include <pmacc/types.hpp>

namespace picongpu
{
    using namespace pmacc;

    template<typename Type_>
    struct Set
    {
        HDINLINE Set(Type_ defaultValue) : value(defaultValue)
        {
        }

        template<typename Dst, typename T_Worker>
        HDINLINE void operator()(T_Worker const&, Dst& dst) const
        {
            dst = value;
        }

    private:
        PMACC_ALIGN(value, const Type_);
    };
} // namespace picongpu

/* Copyright 2013-2024 Felix Schmitt, Heiko Burau, Rene Widera
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

#include "pmacc/dimensions/DataSpace.hpp"
#include "pmacc/memory/boxes/DataBox.hpp"
#include "pmacc/memory/boxes/PitchedBox.hpp"
#include "pmacc/particles/frame_types.hpp"

namespace pmacc
{
    template<class TYPE>
    class VectorDataBox : public DataBox<PitchedBox<TYPE, DIM1>>
    {
    public:
        using BaseType = DataBox<PitchedBox<TYPE, 1U>>;
        using type = TYPE;

        HDINLINE VectorDataBox(TYPE* pointer, const DataSpace<DIM1>& offset = {})
            : BaseType(BaseType(PitchedBox<TYPE, DIM1>(pointer)).shift(offset))
        {
        }

        HDINLINE VectorDataBox() = default;
    };

    /**
     * Specifies a one-dimensional DataBox for more convenient usage.
     *
     * @tparam TYPE type of data represented by the DataBox
     */
    template<class TYPE>
    class TileDataBox : public VectorDataBox<TYPE>
    {
    public:
        using BaseType = VectorDataBox<TYPE>;

        HDINLINE TileDataBox(TYPE* pointer, const DataSpace<DIM1>& offset = DataSpace<DIM1>(0), uint32_t size = 0)
            : BaseType(pointer, offset)
            , size(size)
        {
        }

        /**
         * Returns  size of the Box.
         *
         * @return size of this TileDataBox
         */
        HDINLINE int getSize()
        {
            return size;
        }

        /*object is not  initialized valid, copy a valid instance to this object to get a valid instance*/
        HDINLINE TileDataBox() = default;


    protected:
        PMACC_ALIGN(size, size_t);
    };


} // namespace pmacc

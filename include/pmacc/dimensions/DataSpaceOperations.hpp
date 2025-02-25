/* Copyright 2013-2024 Felix Schmitt, Heiko Burau, Rene Widera,
 *                     Benjamin Worpitz
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
#include "pmacc/math/Vector.hpp"
#include "pmacc/memory/dataTypes/Mask.hpp"
#include "pmacc/types.hpp"

namespace pmacc
{
    /**
     * Implements operations on DataSpace objects such as reduce and extend.
     *
     * @tparam DIM number of dimensions (1-3) of the DataSpace object to operate on
     */
    template<unsigned DIM>
    class DataSpaceOperations
    {
    public:
        /**
         * Reduces the DataSpace object ds of dimension DIM to a DataSpace object of dimension DIM-1.
         * The direction of reduction is passed as an exchange direction in ex.
         *
         * The reduction eliminates the first (and only the first) occuring direction
         * in the test order x-y-z.
         *
         * @param ds DataSpace object to reduce
         * @param ex exchange direction for reduction
         * @return reduced DataSpace with dimension DIM-1
         */
        HDINLINE static DataSpace<DIM - 1> reduce(DataSpace<DIM> ds, uint32_t ex);

        /**
         * Extends the DataSpace object ds of dimension DIM to a DataSpace object of dimension DIM+1.
         * The direction of extension is passed as an exchange direction in ex.
         *
         * The extension maps a element with DIM-position ds into a (DIM+1)-position
         * in a (DIM+1)-dimensional grid. The actual grid to map into
         * is constructed as (target)-(offset).
         *
         * @param ds DataSpace object to extend (e.g. a DIM-position)
         * @param ex exchange direction for extension
         * @param target DataSpace describing size of target grid
         * @param offset DataSpace describing size of target grid's offset
         * @return extended DataSpace with dimension DIM+1
         */
        HDINLINE static DataSpace<DIM + 1> extend(
            DataSpace<DIM> ds,
            uint32_t ex,
            DataSpace<DIM + 1> target,
            DataSpace<DIM + 1> offset);
    };

    template<>
    class DataSpaceOperations<DIM1>
    {
    public:
        HDINLINE static DataSpace<DIM2> extend(
            DataSpace<DIM1> ds,
            uint32_t ex,
            DataSpace<DIM2> target,
            DataSpace<DIM2> offset)
        {
            DataSpace<DIM2> directions = Mask::getRelativeDirections<DIM2>(ex);

            DataSpace<DIM2> result(ds[0], ds[0]);

            // RIGHT
            if(directions.x() == 1)
            {
                result.x() = target.x() - offset.x() - 1;
            }

            // LEFT
            if(directions.x() == -1)
            {
                result.x() = offset.x();
            }

            // TOP
            if(directions.y() == 1)
            {
                result.y() = target.y() - offset.y() - 1;
            }

            // BOTTOM
            if(directions.y() == -1)
            {
                result.y() = offset.y();
            }

            return result;
        }
    };

    template<>
    class DataSpaceOperations<DIM2>
    {
    public:
        HDINLINE static DataSpace<DIM1> reduce(DataSpace<DIM2> ds, uint32_t ex)
        {
            DataSpace<DIM2> directions = Mask::getRelativeDirections<DIM2>(ex);

            if(directions.x() != 0)
                return DataSpace<DIM1>(ds.y());

            if(directions.y() != 0)
                return DataSpace<DIM1>(ds.x());

            return DataSpace<DIM1>(0);
        }

        HDINLINE static DataSpace<DIM3> extend(
            DataSpace<DIM2> ds,
            uint32_t ex,
            DataSpace<DIM3> target,
            DataSpace<DIM3> offset)
        {
            DataSpace<DIM3> directions = Mask::getRelativeDirections<DIM3>(ex);

            DataSpace<DIM3> result;

            const uint32_t x_entry(0);
            const uint32_t z_entry(1);
            uint32_t y_entry(1);

            switch(directions.x())
            {
                // RIGHT
            case 1:
                result.x() = target.x() - offset.x() - 1;
                y_entry = 0;
                break;
                // LEFT
            case -1:
                result.x() = offset.x();
                y_entry = 0;
                break;
            case 0:
                result.x() = ds[x_entry];
                break;
            }

            switch(directions.z())
            {
                // BACK
            case 1:
                result.z() = target.z() - offset.z() - 1;
                break;
                // FRONT
            case -1:
                result.z() = offset.z();
                break;
            case 0:
                result.z() = ds[z_entry];
                break;
            }

            switch(directions.y())
            {
                // BOTTOM
            case 1:
                result.y() = target.y() - offset.y() - 1;
                break;
                // TOP
            case -1:
                result.y() = offset.y();
                break;
            case 0:
                // thsi if fiy lmem usage (old wars result.y()=ds[y_entry] )
                if(y_entry == 0)
                    result.y() = ds.x();
                else
                    result.y() = ds.y();
                break;
            }

            return result;
        }
    };

    template<>
    class DataSpaceOperations<DIM3>
    {
    public:
        HDINLINE static DataSpace<DIM2> reduce(DataSpace<DIM3> ds, uint32_t ex)
        {
            DataSpace<DIM3> directions = Mask::getRelativeDirections<DIM3>(ex);

            if(directions.x() != 0)
                return DataSpace<DIM2>(ds.y(), ds.z());

            if(directions.z() != 0)
                return DataSpace<DIM2>(ds.x(), ds.y());

            if(directions.y() != 0)
                return DataSpace<DIM2>(ds.x(), ds.z());


            return DataSpace<DIM2>(0, 0);
        }
    };
} // namespace pmacc

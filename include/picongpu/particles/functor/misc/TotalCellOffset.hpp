/* Copyright 2017-2024 Rene Widera
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

#include "picongpu/defines.hpp"
#include "picongpu/particles/functor/misc/TotalCellOffset.def"
#include "picongpu/simulation/control/MovingWindow.hpp"


namespace picongpu
{
    namespace particles
    {
        namespace functor
        {
            namespace misc
            {
                struct TotalCellOffset
                {
                    /** constructor
                     *
                     * @param currentStep current simulation time step
                     */
                    HINLINE TotalCellOffset(uint32_t currentStep)
                    {
                        uint32_t const numSlides = MovingWindow::getInstance().getSlideCounter(currentStep);
                        SubGrid<simDim> const& subGrid = Environment<simDim>::get().SubGrid();
                        DataSpace<simDim> const localCells = subGrid.getLocalDomain().size;
                        gpuCellOffsetToTotalOrigin = subGrid.getLocalDomain().offset;
                        gpuCellOffsetToTotalOrigin.y() += numSlides * localCells.y();
                    }

                    /** get cell offset of the supercell
                     *
                     * @tparam T_Worker lockstep worker type
                     *
                     * @param worker lockstep worker
                     * @param localSupercellOffset (in supercells, without any guards) to the
                     *         origin of the local domain
                     */
                    template<typename T_Worker>
                    HDINLINE DataSpace<simDim> operator()(
                        T_Worker const& worker,
                        DataSpace<simDim> const& localSupercellOffset) const
                    {
                        DataSpace<simDim> const superCellToLocalOriginCellOffset(
                            localSupercellOffset * SuperCellSize::toRT());

                        return gpuCellOffsetToTotalOrigin + superCellToLocalOriginCellOffset;
                    }

                private:
                    //! offset in cells to the total domain origin
                    DataSpace<simDim> gpuCellOffsetToTotalOrigin;
                };

            } // namespace misc
        } // namespace functor
    } // namespace particles
} // namespace picongpu

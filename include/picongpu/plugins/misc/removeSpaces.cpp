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

#include "picongpu/plugins/misc/removeSpaces.hpp"

#include <algorithm>
#include <string>


namespace picongpu
{
    namespace plugins
    {
        namespace misc
        {
            std::string removeSpaces(std::string value)
            {
                value.erase(std::remove(value.begin(), value.end(), ' '), value.end());

                return value;
            }
        } // namespace misc
    } // namespace plugins
} // namespace picongpu

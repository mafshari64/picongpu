/* Copyright 2015-2024 Marco Garten
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

/** \file ionizers.hpp
 *
 * Includes containing definition of < Ionization Models >
 * which itself each include their own < Ionization Algorithm >
 * that implements what the model actually DOES
 */

#include "picongpu/particles/ionization/byField/ADK/ADK_Impl.hpp"
#include "picongpu/particles/ionization/byField/BSI/BSI_Impl.hpp"
#include "picongpu/particles/ionization/byField/Keldysh/Keldysh_Impl.hpp"

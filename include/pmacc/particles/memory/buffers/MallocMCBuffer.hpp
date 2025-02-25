/* Copyright 2015-2024 Rene Widera
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

#include "pmacc/alpakaHelper/acc.hpp"
#include "pmacc/dataManagement/ISimulationData.hpp"
#include "pmacc/dimensions/Definition.hpp"

#include <cstdint>
#include <memory>
#include <string>

#if(ALPAKA_ACC_GPU_CUDA_ENABLED || ALPAKA_ACC_GPU_HIP_ENABLED)

#    include <memory>

#    include <mallocMC/mallocMC.hpp>

namespace pmacc
{
    template<typename T_DeviceHeap>
    class MallocMCBuffer : public ISimulationData
    {
    public:
        using DeviceHeap = T_DeviceHeap;
        using BufferType = ::alpaka::Buf<HostDevice, uint8_t, AlpakaDim<DIM1>, MemIdxType>;

        MallocMCBuffer(const std::shared_ptr<DeviceHeap>& deviceHeap);

        virtual ~MallocMCBuffer();

        SimulationDataId getUniqueId() override
        {
            return getName();
        }

        static std::string getName()
        {
            return std::string("MallocMCBuffer");
        }

        int64_t getOffset()
        {
            return hostBufferOffset;
        }

        void synchronize() override;

    private:
        std::optional<BufferType> hostBuffer;
        int64_t hostBufferOffset;
        mallocMC::HeapInfo deviceHeapInfo;
    };


} // namespace pmacc

#    include "pmacc/particles/memory/buffers/MallocMCBuffer.tpp"

#else

namespace pmacc
{
    template<typename T_DeviceHeap>
    class MallocMCBuffer : public ISimulationData
    {
    public:
        MallocMCBuffer(const std::shared_ptr<T_DeviceHeap>&);

        ~MallocMCBuffer() override = default;

        SimulationDataId getUniqueId() override
        {
            return getName();
        }

        static std::string getName()
        {
            return std::string("MallocMCBuffer");
        }

        int64_t getOffset()
        {
            return 0u;
        }

        void synchronize() override
        {
        }
    };

} // namespace pmacc
#endif

/* Copyright 2013-2024 Felix Schmitt, Heiko Burau, Rene Widera,
 *                     Alexander Grund
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

#if(BOOST_LANG_CUDA || BOOST_COMP_HIP)
#    include <mallocMC/mallocMC.hpp>
#endif
#include "pmacc/dimensions/DataSpace.hpp"
#include "pmacc/lockstep.hpp"
#include "pmacc/memory/boxes/DataBox.hpp"
#include "pmacc/memory/boxes/PitchedBox.hpp"
#include "pmacc/particles/Identifier.hpp"
#include "pmacc/particles/frame_types.hpp"
#include "pmacc/particles/memory/dataTypes/FramePointer.hpp"
#include "pmacc/particles/memory/dataTypes/SuperCell.hpp"

#include <pmacc/verify.hpp>

namespace pmacc
{
    /**
     * A DIM-dimensional Box holding frames with particle data.
     *
     * @tparam FRAME datatype for frames
     * @tparam DIM dimension of data (1-3)
     */
    template<class T_Frame, typename T_DeviceHeapHandle, typename T_SuperCellSize, unsigned DIM>
    class ParticlesBox : protected DataBox<PitchedBox<SuperCell<T_Frame, T_SuperCellSize>, DIM>>
    {
    private:
        PMACC_ALIGN(m_deviceHeapHandle, T_DeviceHeapHandle);
        PMACC_ALIGN(hostMemoryOffset, int64_t){0};

    public:
        using FrameType = T_Frame;
        using FramePtr = FramePointer<FrameType>;
        using SuperCellType = SuperCell<FrameType, T_SuperCellSize>;
        using BaseType = DataBox<PitchedBox<SuperCell<FrameType, T_SuperCellSize>, DIM>>;
        using SuperCellSize = T_SuperCellSize;
        using DeviceHeapHandle = T_DeviceHeapHandle;

        static constexpr uint32_t frameSize = FrameType::NumSlots::value;
        static constexpr uint32_t Dim = DIM;

        /** default constructor
         *
         * \warning after this call the object is in a invalid state and must be
         * initialized with an assignment of a valid ParticleBox
         */
        HDINLINE ParticlesBox() = default;

        HDINLINE ParticlesBox(
            const DataBox<PitchedBox<SuperCellType, DIM>>& superCells,
            const DeviceHeapHandle& deviceHeapHandle)
            : BaseType(superCells)
            , m_deviceHeapHandle(deviceHeapHandle)

        {
        }

        HDINLINE ParticlesBox(
            const DataBox<PitchedBox<SuperCellType, DIM>>& superCells,
            const DeviceHeapHandle& deviceHeapHandle,
            int64_t memoryOffset)
            : BaseType(superCells)
            , m_deviceHeapHandle(deviceHeapHandle)
            , hostMemoryOffset(memoryOffset)
        {
        }

        /**
         * Returns an empty frame from data heap.
         *
         * @return an empty frame
         */
        template<typename T_Worker>
        DINLINE FramePtr getEmptyFrame(const T_Worker& worker)
        {
            FrameType* tmp = nullptr;
            const int maxTries = 13; // magic number is not performance critical
            for(int numTries = 0; numTries < maxTries; ++numTries)
            {
#if(BOOST_LANG_CUDA || BOOST_COMP_HIP)
                tmp = (FrameType*) m_deviceHeapHandle.malloc(worker.getAcc(), sizeof(FrameType));
#else
                tmp = new FrameType;
#endif
                if(tmp != nullptr)
                {
                    /* disable all particles since we can not assume that newly allocated memory contains zeros */
                    for(int i = 0; i < static_cast<int>(FrameType::frameSize); ++i)
                        (*tmp)[i][multiMask_] = 0;
                    /* takes care that changed values are visible to all threads inside this block*/
                    alpaka::mem_fence(worker.getAcc(), alpaka::memory_scope::Block{});
                    break;
                }
            }

            PMACC_DEVICE_VERIFY_MSG(tmp != nullptr, "Error: Out of device heap memory in %s:%u\n", __FILE__, __LINE__);

            return FramePtr(tmp);
        }

        /**
         * Removes frame from heap data heap.
         *
         * @param frame frame to remove
         */
        template<typename T_Worker>
        DINLINE void removeFrame(const T_Worker& worker, FramePtr& frame)
        {
#if(BOOST_LANG_CUDA || BOOST_COMP_HIP)
            m_deviceHeapHandle.free(worker.getAcc(), (void*) frame.ptr);
#else
            delete(frame.ptr);
#endif
            frame.ptr = nullptr;
        }

        HDINLINE
        FramePtr mapPtr(FramePtr devPtr) const
        {
#if(PMACC_DEVICE_COMPILE == 1)
            return devPtr;
#else
            int64_t useOffset = hostMemoryOffset * static_cast<int64_t>(devPtr.ptr != nullptr);
            return FramePtr(reinterpret_cast<FrameType*>(reinterpret_cast<char*>(devPtr.ptr) - useOffset));
#endif
        }

        /**
         * Returns the next frame in the linked list.
         *
         * @param frame the active frame
         * @return the next frame in the list
         */
        HDINLINE FramePtr getNextFrame(const FramePtr& frame) const
        {
            return mapPtr(frame->nextFrame.ptr);
        }

        /**
         * Returns the previous frame in the linked list.
         *
         * @param frame the active frame
         * @return the previous frame in the list
         */
        HDINLINE FramePtr getPreviousFrame(const FramePtr& frame) const
        {
            return mapPtr(frame->previousFrame.ptr);
        }

        /**
         * Returns the last frame of a supercell.
         *
         * @param idx position of supercell
         * @return the last frame of the linked list from supercell
         */
        HDINLINE FramePtr getLastFrame(const DataSpace<DIM>& idx) const
        {
            return mapPtr(getSuperCell(idx).LastFramePtr());
        }

        /**
         * Returns the first frame of a supercell.
         *
         * @param idx position of supercell
         * @return the first frame of the linked list from supercell
         */
        HDINLINE FramePtr getFirstFrame(const DataSpace<DIM>& idx) const
        {
            return mapPtr(getSuperCell(idx).FirstFramePtr());
        }

        /**
         * Sets frame as the first frame of a supercell.
         *
         * @param frame frame to set as first frame
         * @param idx position of supercell
         */
        template<typename T_Worker>
        DINLINE void setAsFirstFrame(T_Worker const& worker, FramePtr& frame, DataSpace<DIM> const& idx)
        {
            FrameType** firstFrameNativPtr = &(getSuperCell(idx).firstFramePtr);

            frame->previousFrame = FramePtr();
            frame->nextFrame = FramePtr(*firstFrameNativPtr);

            /* - takes care that `next[index]` is visible to all threads on the gpu
             * - this is needed because later on in this method we change `previous`
             *   of an other frame, this must be done in order!
             */
            alpaka::mem_fence(worker.getAcc(), alpaka::memory_scope::Device{});

            FramePtr oldFirstFramePtr((FrameType*) alpaka::atomicExch(
                worker.getAcc(),
                (unsigned long long int*) firstFrameNativPtr,
                (unsigned long long int) frame.ptr,
                ::alpaka::hierarchy::Grids{}));

            frame->nextFrame = oldFirstFramePtr;
            if(oldFirstFramePtr.isValid())
            {
                oldFirstFramePtr->previousFrame = frame;
            }
            else
            {
                // we add the first frame in supercell
                getSuperCell(idx).lastFramePtr = frame.ptr;
            }
        }

        /**
         * Sets frame as the last frame of a supercell.
         *
         * @param frame frame to set as last frame
         * @param idx position of supercell
         */
        template<typename T_Worker>
        DINLINE void setAsLastFrame(T_Worker const& worker, FramePointer<FrameType>& frame, DataSpace<DIM> const& idx)
        {
            FrameType** lastFrameNativPtr = &(getSuperCell(idx).lastFramePtr);

            frame->nextFrame = FramePtr();
            frame->previousFrame = FramePtr(*lastFrameNativPtr);

            /* - takes care that `next[index]` is visible to all threads on the gpu
             * - this is needed because later on in this method we change `next`
             *   of an other frame, this must be done in order!
             */
            alpaka::mem_fence(worker.getAcc(), alpaka::memory_scope::Device{});

            FramePtr oldLastFramePtr((FrameType*) alpaka::atomicExch(
                worker.getAcc(),
                (unsigned long long int*) lastFrameNativPtr,
                (unsigned long long int) frame.ptr,
                ::alpaka::hierarchy::Grids{}));

            frame->previousFrame = oldLastFramePtr;
            if(oldLastFramePtr.isValid())
            {
                oldLastFramePtr->nextFrame = frame;
            }
            else
            {
                // we add the first frame in supercell
                getSuperCell(idx).firstFramePtr = frame.ptr;
            }
        }

        /**
         * Removes the last frame of a supercell.
         * This call is not threadsave, only one thread from a supercell may call this function.
         * @param idx position of supercell
         * @return true if more frames in list, else false
         */
        template<typename T_Worker>
        DINLINE bool removeLastFrame(const T_Worker& worker, const DataSpace<DIM>& idx)
        {
            //!\todo this is not thread save
            FrameType** lastFrameNativPtr = &(getSuperCell(idx).lastFramePtr);

            FramePtr last(*lastFrameNativPtr);
            if(last.isValid())
            {
                FramePtr prev(last->previousFrame);

                if(prev.isValid())
                {
                    prev->nextFrame = FramePtr(); // set to invalid frame
                    *lastFrameNativPtr = prev.ptr; // set new last frame
                    removeFrame(worker, last);
                    return true;
                }
                // remove last frame of supercell
                getSuperCell(idx).firstFramePtr = nullptr;
                getSuperCell(idx).lastFramePtr = nullptr;

                removeFrame(worker, last);
            }
            return false;
        }

        HDINLINE decltype(auto) getSuperCell(DataSpace<DIM> idx) const
        {
            return BaseType::operator()(idx);
        }

        HDINLINE decltype(auto) getSuperCell(DataSpace<DIM> idx)
        {
            return BaseType::operator()(idx);
        }
    };

    namespace lockstep::traits
    {
        //! Specialization to create a lockstep block configuration out of a particle buffer.
        template<class T_Frame, typename T_DeviceHeapHandle, typename T_SuperCellSize, unsigned T_dim>
        struct MakeBlockCfg<ParticlesBox<T_Frame, T_DeviceHeapHandle, T_SuperCellSize, T_dim>> : std::true_type
        {
            static constexpr uint32_t frameSize
                = ParticlesBox<T_Frame, T_DeviceHeapHandle, T_SuperCellSize, T_dim>::FrameType::frameSize;
            using type = BlockCfg<math::CT::UInt32<frameSize>>;
        };
    } // namespace lockstep::traits

} // namespace pmacc

9.7.9.27.1.2. Data Movement and Conversion Instructions: cp.async.bulk.tensor
cp.async.bulk.tensor

Initiates an asynchronous copy operation on the tensor data from one state space to another.

Syntax

// global -> shared::cta
cp.async.bulk.tensor.dim.dst.src{.load_mode}.completion_mechanism{.cta_group}{.level::cache_hint}
                                   [dstMem], [tensorMap, tensorCoords], [mbar]{, im2colInfo} {, cache-policy}

.dst =                  { .shared::cta }
.src =                  { .global }
.dim =                  { .1d, .2d, .3d, .4d, .5d }
.completion_mechanism = { .mbarrier::complete_tx::bytes }
.cta_group =            { .cta_group::1, .cta_group::2 }
.load_mode =            { .tile, .tile::gather4, .im2col, .im2col::w, .im2col::w::128 }
.level::cache_hint =    { .L2::cache_hint }


// global -> shared::cluster
cp.async.bulk.tensor.dim.dst.src{.load_mode}.completion_mechanism{.multicast}{.cta_group}{.level::cache_hint}
                                   [dstMem], [tensorMap, tensorCoords], [mbar]{, im2colInfo}
                                   {, ctaMask} {, cache-policy}

.dst =                  { .shared::cluster }
.src =                  { .global }
.dim =                  { .1d, .2d, .3d, .4d, .5d }
.completion_mechanism = { .mbarrier::complete_tx::bytes }
.cta_group =            { .cta_group::1, .cta_group::2 }
.load_mode =            { .tile, .tile::gather4, .im2col, .im2col::w, .im2col::w::128 }
.level::cache_hint =    { .L2::cache_hint }
.multicast =            { .multicast::cluster  }


// shared::cta -> global
cp.async.bulk.tensor.dim.dst.src{.load_mode}.completion_mechanism{.level::cache_hint}
                                   [tensorMap, tensorCoords], [srcMem] {, cache-policy}

.dst =                  { .global }
.src =                  { .shared::cta }
.dim =                  { .1d, .2d, .3d, .4d, .5d }
.completion_mechanism = { .bulk_group }
.load_mode =            { .tile, .tile::scatter4, .im2col_no_offs }
.level::cache_hint =    { .L2::cache_hint }
Description

cp.async.bulk.tensor is a non-blocking instruction which initiates an asynchronous copy operation of tensor data from the location in .src state space to the location in the .dst state space.

The operand dstMem specifies the location in the .dst state space into which the tensor data has to be copied and srcMem specifies the location in the .src state space from which the tensor data has to be copied.

When .dst is specified as .shared::cta, the address dstMem must be in the shared memory of the executing CTA within the cluster, otherwise the behavior is undefined.

When .dst is specified as .shared::cluster, the address dstMem can be in the shared memory of any of the CTAs within the current cluster.

The operand tensorMap is the generic address of the opaque tensor-map object which resides in .param space or .const space or .global space. The operand tensorMap specifies the properties of the tensor copy operation, as described in Tensor-map. The tensorMap is accessed in tensormap proxy. Refer to the CUDA programming guide for creating the tensor-map objects on the host side.

The dimension of the tensor data is specified by the .dim modifier.

The vector operand tensorCoords specifies the starting coordinates in the tensor data in the global memory from or to which the copy operation has to be performed. The individual tensor coordinates in tensorCoords are of type .s32. The format of vector argument tensorCoords is dependent on .load_mode specified and is as follows:

.load_mode

tensorCoords

Semantics

.tile::scatter4

{col_idx, row_idx0, row_idx1, row_idx2, row_idx3}

Fixed length vector of size 5. The five elements together specify the start co-ordinates of the four rows.

.tile::gather4

Rest all

{d0, .., dn} for n = .dim

Vector of n elements where n = .dim. The elements indicate the offset in each of the dimension.

The modifier .completion_mechanism specifies the completion mechanism that is supported on the instruction variant. The completion mechanisms that are supported for different variants are summarized in the following table:

.completion-mechanism

.dst

.src

Completion mechanism

Needed for completion of entire Async operation

optionally can be used for the completion of reading of the tensormap object

.mbarrier::...

.shared::cta

.global

mbarrier based

Bulk async-group based

.shared::cluster

.global

.bulk_group

.global

.shared::cta

Bulk async-group based

The modifier .mbarrier::complete_tx::bytes specifies that the cp.async.bulk.tensor variant uses mbarrier based completion mechanism. Upon the completion of the asynchronous copy operation, the complete-tx operation, with completeCount argument equal to amount of data copied in bytes, will be performed on the mbarrier object specified by the operand mbar. This instruction accesses its mbarrier operand using generic-proxy.

The modifier .cta_group can only be specified with the mbarrier based completion mechanism. The modifier .cta_group is used to signal either the odd numbered CTA or the even numbered CTA among the CTA-Pair. When .cta_group::1 is specified, the mbarrier object mbar that is specified must be in the shared memory of the same CTA as the shared memory destination dstMem. When .cta_group::2 is specified, the mbarrier object mbar can be in shared memory of either the same CTA as the shared memory destination dstMem or in its peer-CTA. If .cta_group is not specified, then it defaults to .cta_group::1.

The modifier .bulk_group specifies that the cp.async.bulk.tensor variant uses bulk async-group based completion mechanism.

The qualifier .load_mode specifies how the data in the source location is copied into the destination location. If .load_mode is not specified, it defaults to .tile.

In .tile mode, the multi-dimensional layout of the source tensor is preserved at the destination. In .tile::gather4 mode, four rows in 2-dimnesional source tensor are combined to form a single 2-dimensional destination tensor. In .tile::scatter4 mode, single 2-dimensional source tensor is divided into four rows in 2-dimensional destination tensor. Details of .tile::scatter4/.tile::gather4 modes are described in .tile::scatter4 and .tile::gather4 modes.

In .im2col and .im2col::* modes, some dimensions of the source tensors are unrolled in a single dimensional column at the destination. Details of the im2col and .im2col::* modes are described in im2col mode and im2col::w and im2col::w::128 modes respectively. In .im2col and .im2col::* modes, the tensor has to be at least 3-dimensional. The vector operand im2colInfo can be specified only when .load_mode is .im2col or .im2col::w or .im2col::w::128. The format of the vector argument im2colInfo is dependent on the exact im2col mode and is as follows:

Exact im2col mode

im2colInfo argument

Semantics

.im2col

{ i2cOffW , i2cOffH , i2cOffD } for .dim = .5d

A vector of im2col offsets whose vector size is two less than number of dimensions .dim.

.im2col::w

{ wHalo, wOffset }

A vector of 2 arguments containing wHalo and wOffset arguments.

.im2col::w::128

.im2col_no_offs

im2colInfo is not applicable.

im2colInfo is not applicable.

Argument wHalo is a 16bit unsigned integer whose valid set of values differs on the load-mode and is as follows: - Im2col::w mode : valid range is [0, 512). - Im2col::w::128 mode : valid range is [0, 32).

Argument wOffset is a 16bit unsigned integer whose valid range of values is [0, 32).

The optional modifier .multicast::cluster allows copying of data from global memory to shared memory of multiple CTAs in the cluster. Operand ctaMask specifies the destination CTAs in the cluster such that each bit position in the 16-bit ctaMask operand corresponds to the %ctaid of the destination CTA. The source data is multicast to the same offset as dstMem in the shared memory of each destination CTA. When .cta_group is specified as:

.cta_group::1 : The mbarrier signal is also multicasted to the same offset as mbar in the shared memory of the destination CTA.

.cta_group::2 : The mbarrier signal is multicasted either to all the odd numbered CTAs or the even numbered CTAs within the corresponding CTA-Pair. For each destination CTA specified in the ctaMask, the mbarrier signal is sent either to the destination CTA or its peer-CTA based on CTAs %cluster_ctarank parity of shared memory where the mbarrier object mbar resides.

When the optional argument cache-policy is specified, the qualifier .level::cache_hint is required. The 64-bit operand cache-policy specifies the cache eviction policy that may be used during the memory access.

cache-policy is a hint to the cache subsystem and may not always be respected. It is treated as a performance hint only, and does not change the memory consistency behavior of the program.

The copy operation in cp.async.bulk.tensor is treated as a weak memory operation and the complete-tx operation on the mbarrier has .release semantics at the .cluster scope as described in the Memory Consistency Model.

Notes

.multicast::cluster qualifier is optimized for target architecture sm_90a/sm_100f/sm_100a/ sm_103f/sm_103a/sm_110f/sm_110a and may have substantially reduced performance on other targets and hence .multicast::cluster is advised to be used with .target sm_90a/sm_100f/ sm_100a/sm_103f/sm_103a/sm_110f/sm_110a.

PTX ISA Notes

Introduced in PTX ISA version 8.0.

Support for .shared::cta as destination state space is introduced in PTX ISA version 8.6.

Support for qualifiers .tile::gather4 and .tile::scatter4 introduced in PTX ISA version 8.6.

Support for qualifiers .im2col::w and .im2col::w::128 introduced in PTX ISA version 8.6.

Support for qualifier .cta_group introduced in PTX ISA version 8.6.

Target ISA Notes

Requires sm_90 or higher.

.multicast::cluster qualifier advised to be used with .target sm_90a or sm_100f or sm_100a or sm_103f or sm_103a or sm_110f or sm_110a.

Qualifiers .tile::gather4 and .im2col::w require:

sm_100a when destination state space is .shared::cluster and is supported on sm_100f from PTX ISA version 8.8.

sm_100 or higher when destination state space is .shared::cta.

Qualifier .tile::scatter4 is supported on following architectures:

sm_100a

sm_101a (Renamed to sm_110a from PTX ISA version 9.0)

And is supported on following family-specific architectures from PTX ISA version 8.8:

sm_100f or higher in the same family

sm_101f or higher in the same family (Renamed to sm_110f from PTX ISA version 9.0)

sm_110f or higher in the same family

Qualifier .im2col::w::128 is supported on following architectures:

sm_100a

sm_101a (Renamed to sm_110a from PTX ISA version 9.0)

And is supported on following family-specific architectures from PTX ISA version 8.8:

sm_100f or higher in the same family

sm_101f or higher in the same family (Renamed to sm_110f from PTX ISA version 9.0)

sm_110f or higher in the same family

Qualifier .cta_group is supported on following architectures:

sm_100a

sm_101a (Renamed to sm_110a from PTX ISA version 9.0)

And is supported on following family-specific architectures from PTX ISA version 8.8:

sm_100f or higher in the same family

sm_101f or higher in the same family (Renamed to sm_110f from PTX ISA version 9.0)

sm_110f or higher in the same family

Examples

.reg .b16 ctaMask;
.reg .u16 i2cOffW, i2cOffH, i2cOffD;
.reg .b64 l2CachePolicy;

cp.async.bulk.tensor.1d.shared::cta.global.mbarrier::complete_tx::bytes.tile  [sMem0], [tensorMap0, {tc0}], [mbar0];

@p cp.async.bulk.tensor.5d.shared::cta.global.im2col.mbarrier::complete_tx::bytes
                     [sMem2], [tensorMap2, {tc0, tc1, tc2, tc3, tc4}], [mbar2], {i2cOffW, i2cOffH, i2cOffD};

cp.async.bulk.tensor.1d.shared::cluster.global.mbarrier::complete_tx::bytes.tile  [sMem0], [tensorMap0, {tc0}], [mbar0];

@p cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster
                     [sMem1], [tensorMap1, {tc0, tc1}], [mbar2], ctaMask;

@p cp.async.bulk.tensor.5d.shared::cluster.global.im2col.mbarrier::complete_tx::bytes
                     [sMem2], [tensorMap2, {tc0, tc1, tc2, tc3, tc4}], [mbar2], {i2cOffW, i2cOffH, i2cOffD};

@p cp.async.bulk.tensor.3d.im2col.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint
                     [sMem3], [tensorMap3, {tc0, tc1, tc2}], [mbar3], {i2cOffW}, policy;

@p cp.async.bulk.tensor.1d.global.shared::cta.bulk_group  [tensorMap3, {tc0}], [sMem3];

cp.async.bulk.tensor.2d.tile::gather4.shared::cluster.global.mbarrier::complete_tx::bytes
                     [sMem5], [tensorMap6, {x0, y0, y1, y2, y3}], [mbar5];

cp.async.bulk.tensor.3d.im2col::w.shared::cluster.global.mbarrier::complete_tx::bytes
                     [sMem4], [tensorMap5, {t0, t1, t2}], [mbar4], {im2colwHalo, im2colOff};

cp.async.bulk.tensor.1d.shared::cluster.global.tile.cta_group::2
                     [sMem6], [tensorMap7, {tc0}], [peerMbar];

-------

CUresult cuTensorMapEncodeTiled ( CUtensorMap* tensorMap, CUtensorMapDataType tensorDataType, cuuint32_t tensorRank, void* globalAddress, const cuuint64_t* globalDim, const cuuint64_t* globalStrides, const cuuint32_t* boxDim, const cuuint32_t* elementStrides, CUtensorMapInterleave interleave, CUtensorMapSwizzle swizzle, CUtensorMapL2promotion l2Promotion, CUtensorMapFloatOOBfill oobFill )
Create a tensor map descriptor object representing tiled memory region.
Parameters
tensorMap
- Tensor map object to create
tensorDataType
- Tensor data type
tensorRank
- Dimensionality of tensor
globalAddress
- Starting address of memory region described by tensor
globalDim
- Array containing tensor size (number of elements) along each of the tensorRank dimensions
globalStrides
- Array containing stride size (in bytes) along each of the tensorRank - 1 dimensions
boxDim
- Array containing traversal box size (number of elments) along each of the tensorRank dimensions. Specifies how many elements to be traversed along each tensor dimension.
elementStrides
- Array containing traversal stride in each of the tensorRank dimensions
interleave
- Type of interleaved layout the tensor addresses
swizzle
- Bank swizzling pattern inside shared memory
l2Promotion
- L2 promotion size
oobFill
- Indicate whether zero or special NaN constant must be used to fill out-of-bound elements
Returns
CUDA_SUCCESS, CUDA_ERROR_DEINITIALIZED, CUDA_ERROR_NOT_INITIALIZED, CUDA_ERROR_INVALID_CONTEXT, CUDA_ERROR_INVALID_VALUE

Description
Creates a descriptor for Tensor Memory Access (TMA) object specified by the parameters describing a tiled region and returns it in tensorMap.

Tensor map objects are only supported on devices of compute capability 9.0 or higher. Additionally, a tensor map object is an opaque value, and, as such, should only be accessed through CUDA APIs and PTX.

The parameters passed are bound to the following requirements:


tensorMap address must be aligned to 64 bytes.


tensorDataType has to be an enum from CUtensorMapDataType which is defined as:
‎    typedef enum CUtensorMapDataType_enum {
              CU_TENSOR_MAP_DATA_TYPE_UINT8 = 0,       // 1 byte
              CU_TENSOR_MAP_DATA_TYPE_UINT16,          // 2 bytes
              CU_TENSOR_MAP_DATA_TYPE_UINT32,          // 4 bytes
              CU_TENSOR_MAP_DATA_TYPE_INT32,           // 4 bytes
              CU_TENSOR_MAP_DATA_TYPE_UINT64,          // 8 bytes
              CU_TENSOR_MAP_DATA_TYPE_INT64,           // 8 bytes
              CU_TENSOR_MAP_DATA_TYPE_FLOAT16,         // 2 bytes
              CU_TENSOR_MAP_DATA_TYPE_FLOAT32,         // 4 bytes
              CU_TENSOR_MAP_DATA_TYPE_FLOAT64,         // 8 bytes
              CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,        // 2 bytes
              CU_TENSOR_MAP_DATA_TYPE_FLOAT32_FTZ,     // 4 bytes
              CU_TENSOR_MAP_DATA_TYPE_TFLOAT32,        // 4 bytes
              CU_TENSOR_MAP_DATA_TYPE_TFLOAT32_FTZ,    // 4 bytes
              CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B,    // 4 bits
              CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN16B,   // 4 bits
              CU_TENSOR_MAP_DATA_TYPE_16U6_ALIGN16B    // 6 bits
          } CUtensorMapDataType;
CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B copies '16 x U4' packed values to memory aligned as 8 bytes. There are no gaps between packed values. CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN16B copies '16 x U4' packed values to memory aligned as 16 bytes. There are 8 byte gaps between every 8 byte chunk of packed values. CU_TENSOR_MAP_DATA_TYPE_16U6_ALIGN16B copies '16 x U6' packed values to memory aligned as 16 bytes. There are 4 byte gaps between every 12 byte chunk of packed values.

tensorRank must be non-zero and less than or equal to the maximum supported dimensionality of 5. If interleave is not CU_TENSOR_MAP_INTERLEAVE_NONE, then tensorRank must additionally be greater than or equal to 3.


globalAddress, which specifies the starting address of the memory region described, must be 16 byte aligned. The following requirements need to also be met:
When interleave is CU_TENSOR_MAP_INTERLEAVE_32B, globalAddress must be 32 byte aligned.

When tensorDataType is CU_TENSOR_MAP_DATA_TYPE_16U6_ALIGN16B or CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN16B, globalAddress must be 32 byte aligned.


globalDim array, which specifies tensor size of each of the tensorRank dimensions, must be non-zero and less than or equal to 2^32. Additionally, the following requirements need to be met for the packed data types:
When tensorDataType is CU_TENSOR_MAP_DATA_TYPE_16U6_ALIGN16B or CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN16B, globalDim[0] must be a multiple of 128.

When tensorDataType is CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B, globalDim[0] must be a multiple of 2.

Dimension for the packed data types must reflect the number of individual U# values.


globalStrides array, which specifies tensor stride of each of the lower tensorRank - 1 dimensions in bytes, must be a multiple of 16 and less than 2^40. Additionally, the following requirements need to be met:
When interleave is CU_TENSOR_MAP_INTERLEAVE_32B, the strides must be a multiple of 32.

When tensorDataType is CU_TENSOR_MAP_DATA_TYPE_16U6_ALIGN16B or CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN16B, the strides must be a multiple of 32. Each following dimension specified includes previous dimension stride:
‎    globalStrides[0] = globalDim[0] * elementSizeInBytes(tensorDataType) + padding[0];
          for (i = 1; i < tensorRank - 1; i++)
              globalStrides[i] = globalStrides[i – 1] * (globalDim[i] + padding[i]);
              assert(globalStrides[i] >= globalDim[i]);

boxDim array, which specifies number of elements to be traversed along each of the tensorRank dimensions, must be non-zero and less than or equal to 256. Additionally, the following requirements need to be met:
When interleave is CU_TENSOR_MAP_INTERLEAVE_NONE, { boxDim[0] * elementSizeInBytes( tensorDataType ) } must be a multiple of 16 bytes.

When tensorDataType is CU_TENSOR_MAP_DATA_TYPE_16U6_ALIGN16B or CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN16B, boxDim[0] must be 128.


elementStrides array, which specifies the iteration step along each of the tensorRank dimensions, must be non-zero and less than or equal to 8. Note that when interleave is CU_TENSOR_MAP_INTERLEAVE_NONE, the first element of this array is ignored since TMA doesn’t support the stride for dimension zero. When all elements of elementStrides array is one, boxDim specifies the number of elements to load. However, if the elementStrides[i] is not equal to one, then TMA loads ceil( boxDim[i] / elementStrides[i]) number of elements along i-th dimension. To load N elements along i-th dimension, boxDim[i] must be set to N * elementStrides[i].


interleave specifies the interleaved layout of type CUtensorMapInterleave, which is defined as:
‎    typedef enum CUtensorMapInterleave_enum {
              CU_TENSOR_MAP_INTERLEAVE_NONE = 0,
              CU_TENSOR_MAP_INTERLEAVE_16B,
              CU_TENSOR_MAP_INTERLEAVE_32B
          } CUtensorMapInterleave;
TMA supports interleaved layouts like NC/8HWC8 where C8 utilizes 16 bytes in memory assuming 2 byte per channel or NC/16HWC16 where C16 uses 32 bytes. When interleave is CU_TENSOR_MAP_INTERLEAVE_NONE and swizzle is not CU_TENSOR_MAP_SWIZZLE_NONE, the bounding box inner dimension (computed as boxDim[0] multiplied by element size derived from tensorDataType) must be less than or equal to the swizzle size.
CU_TENSOR_MAP_SWIZZLE_32B requires the bounding box inner dimension to be <= 32.

CU_TENSOR_MAP_SWIZZLE_64B requires the bounding box inner dimension to be <= 64.

CU_TENSOR_MAP_SWIZZLE_128B* require the bounding box inner dimension to be <= 128. Additionally, tensorDataType of CU_TENSOR_MAP_DATA_TYPE_16U6_ALIGN16B requires interleave to be CU_TENSOR_MAP_INTERLEAVE_NONE.


swizzle, which specifies the shared memory bank swizzling pattern, has to be of type CUtensorMapSwizzle which is defined as:
‎    typedef enum CUtensorMapSwizzle_enum {
              CU_TENSOR_MAP_SWIZZLE_NONE = 0,
              CU_TENSOR_MAP_SWIZZLE_32B,                   // Swizzle 16B chunks within 32B  span
              CU_TENSOR_MAP_SWIZZLE_64B,                   // Swizzle 16B chunks within 64B  span
              CU_TENSOR_MAP_SWIZZLE_128B,                  // Swizzle 16B chunks within 128B span
              CU_TENSOR_MAP_SWIZZLE_128B_ATOM_32B,         // Swizzle 32B chunks within 128B span
              CU_TENSOR_MAP_SWIZZLE_128B_ATOM_32B_FLIP_8B, // Swizzle 32B chunks within 128B span, additionally swap lower 8B with upper 8B within each 16B for every alternate row
              CU_TENSOR_MAP_SWIZZLE_128B_ATOM_64B          // Swizzle 64B chunks within 128B span
          } CUtensorMapSwizzle;
Data are organized in a specific order in global memory; however, this may not match the order in which the application accesses data in shared memory. This difference in data organization may cause bank conflicts when shared memory is accessed. In order to avoid this problem, data can be loaded to shared memory with shuffling across shared memory banks. When interleave is CU_TENSOR_MAP_INTERLEAVE_32B, swizzle must be CU_TENSOR_MAP_SWIZZLE_32B. Other interleave modes can have any swizzling pattern. When the tensorDataType is CU_TENSOR_MAP_DATA_TYPE_16U6_ALIGN16B, only the following swizzle modes are supported:
CU_TENSOR_MAP_SWIZZLE_NONE (Load & Store)

CU_TENSOR_MAP_SWIZZLE_128B (Load & Store)

CU_TENSOR_MAP_SWIZZLE_128B_ATOM_32B (Load & Store)

CU_TENSOR_MAP_SWIZZLE_128B_ATOM_64B (Store only) When the tensorDataType is CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN16B, only the following swizzle modes are supported:

CU_TENSOR_MAP_SWIZZLE_NONE (Load only)

CU_TENSOR_MAP_SWIZZLE_128B (Load only)

CU_TENSOR_MAP_SWIZZLE_128B_ATOM_32B (Load only)


l2Promotion specifies L2 fetch size which indicates the byte granurality at which L2 requests is filled from DRAM. It must be of type CUtensorMapL2promotion, which is defined as:
‎    typedef enum CUtensorMapL2promotion_enum {
              CU_TENSOR_MAP_L2_PROMOTION_NONE = 0,
              CU_TENSOR_MAP_L2_PROMOTION_L2_64B,
              CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
              CU_TENSOR_MAP_L2_PROMOTION_L2_256B
          } CUtensorMapL2promotion;

oobFill, which indicates whether zero or a special NaN constant should be used to fill out-of-bound elements, must be of type CUtensorMapFloatOOBfill which is defined as:
‎    typedef enum CUtensorMapFloatOOBfill_enum {
              CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE = 0,
              CU_TENSOR_MAP_FLOAT_OOB_FILL_NAN_REQUEST_ZERO_FMA
          } CUtensorMapFloatOOBfill;
Note that CU_TENSOR_MAP_FLOAT_OOB_FILL_NAN_REQUEST_ZERO_FMA can only be used when tensorDataType represents a floating-point data type, and when tensorDataType is not CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B, CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN16B, and CU_TENSOR_MAP_DATA_TYPE_16U6_ALIGN16B.
See also:

cuTensorMapEncodeIm2col, cuTensorMapEncodeIm2colWide, cuTensorMapReplaceAddress
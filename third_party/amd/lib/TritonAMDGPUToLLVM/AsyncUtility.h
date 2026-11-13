#ifndef TRITON_THIRD_PARTY_AMD_LIB_TRITONAMDGPUTOLLVM_ASYNCUTILITY_H_
#define TRITON_THIRD_PARTY_AMD_LIB_TRITONAMDGPUTOLLVM_ASYNCUTILITY_H_

#include "mlir/Dialect/LLVMIR/LLVMInterfaces.h"
#include "mlir/IR/BuiltinOps.h"
#include "triton/Conversion/MLIRTypes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

namespace mlir::triton::AMD {
class TargetInfo;
// Annotates LocalLoadOps with ttg.amdg.syncedByAsyncWait=true if they are
// synced by an AsyncWait.
void annotateLocalLoadsSyncedViaAsyncWait(ModuleOp mod);

// Getter for the annotation applied by annotateLocalLoadsSyncedViaAsyncWait
bool isSyncedViaAsyncWait(Operation *localLoadOp);

// LLVM is unable to deduce dependencies across warps and loop iterations for
// AsyncCopy and LocalLoad and will emit conservative wait counts. In triton the
// dependency is models via AsyncWait, e.g.
//   %token1 = ttg.async_copy_global_to_local/amdg.buffer_load_to_local
//   %token2 = ttg.async_wait %token1
//   %1      = ttg.local_load .. token %token2
// For such cases AsyncWait will emit the correct wait and the conservative
// waits are redundant and hindering performance/interleaving.
// To disable the conservative waits two alias scopes are created:
//   1) "amdg.AsyncCopies" will contain all AsyncCopy ops
//   2) "amdg.LocalLoad" will contain all LocalLoads manually synchronized via
//      AsyncWait
// ALl manually synchronized LocalLoads will additionally have "AsyncCopies" as
// a non alias scope to disable the implicit waits from the LLVM backend

// If localLoadOp has a token from an AsyncWait:
//  - Attaches "amdg.LocalLoad" alias scope to llLoadOp
//  - Attaches "amdg.AsyncCopies" as *non* alias scope to llLoadOp
void addLocalLoadNoAliasScope(Operation *localLoadOp,
                              LLVM::AliasAnalysisOpInterface llLoadOp);
// Overload from above without checking the AsyncToken
void addLocalLoadNoAliasScope(LLVM::AliasAnalysisOpInterface llLoadOp);
// Attaches the "AsyncCopies" alias scope to llLoadDirectToLdsOp
void addAsyncCopyAliasScope(LLVM::AliasAnalysisOpInterface llLoadDirectToLdsOp);

// Async loads into LDS mask redundant threads either by branching around the
// load or by giving them an out-of-range LDS address, which the hardware drops.
// With the address form every warp still issues the load, including a warp
// whose data is entirely redundant, so wait counts must count it for all warps.
// The global load can mask per lane through the address on targets with
// direct-to-LDS scatter; otherwise only a warp-uniform predicate without a
// per-lane mask (or `other` values for buffer loads) uses the address form.
bool globalLoadToLdsMasksViaLdsAddress(bool supportsDirectToLdsScatter,
                                       bool threadPredIsWarpUniform,
                                       bool hasMask);
bool bufferLoadToLdsMasksViaLdsAddress(bool threadPredIsWarpUniform,
                                       bool hasOther);

// Finds the largest supported vecSize smaller than maxVecSize. Returns 0 if
// there is none
unsigned
fitToValidDirectToLdsVecSize(unsigned maxVecSize, unsigned elemBitwidth,
                             const triton::AMD::TargetInfo &targetInfo);

} // namespace mlir::triton::AMD

#endif // TRITON_THIRD_PARTY_AMD_LIB_TRITONAMDGPUTOLLVM_ASYNCUTILITY_H_

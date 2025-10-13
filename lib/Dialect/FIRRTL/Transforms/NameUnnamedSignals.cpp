//===- NameUnnamedSignals.cpp - Add wires for unnamed signals ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the NameUnnamedSignals pass. This pass traverses the FIRRTL
// circuit and adds a wire for every signal that doesn't have a name.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/FIRRTL/FIRRTLAnnotations.h"
#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "circt/Dialect/FIRRTL/Passes.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/raw_ostream.h"

namespace circt {
namespace firrtl {
#define GEN_PASS_DEF_NAMEUNNAMEDSIGNALS
#include "circt/Dialect/FIRRTL/Passes.h.inc"
} // namespace firrtl
} // namespace circt

using namespace circt;
using namespace firrtl;

namespace {
struct NameUnnamedSignalsPass
    : public impl::NameUnnamedSignalsBase<NameUnnamedSignalsPass> {

  void runOnOperation() override;
};
} // end anonymous namespace

/// The main logic for the NameUnnamedSignals pass.
void NameUnnamedSignalsPass::runOnOperation() {
  auto circuitOp = getOperation();

  int i = 0;

  // Traverse all modules in the circuit.
  for (auto module : circuitOp.getOps<FModuleOp>()) {
    OpBuilder builder(module.getBodyRegion());

    // Traverse all operations in the module.
    module.walk([&](Operation *op) {
      // Check if the operation has a result without a name.

      // op->dumpPretty();

      for (OpResult result : op->getResults()) {
        if (!result.hasOneUse() || !llvm::dyn_cast<firrtl::FIRRTLType>(result.getType()))
          continue;

        // Generate a unique name for the wire.
        std::string wireName = "anon_" + std::to_string(i++);

        // Insert a wire for the unnamed signal.
        builder.setInsertionPointAfter(op);
        auto wire = builder.create<WireOp>(op->getLoc(), llvm::dyn_cast<firrtl::FIRRTLType>(result.getType()),
                                           builder.getStringAttr(wireName));

        AnnotationSet::addDontTouch(wire);

        // Replace all uses of the unnamed signal with the new wire.
        result.replaceAllUsesWith(wire->getResult(0));

        builder.create<MatchingConnectOp>(op->getLoc(), wire->getResult(0), result);

      }
    });
  }
}
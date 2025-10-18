//===- HWPowerAnalyze.cpp - Analyze a HW circuit pwr and area ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/HW/PortImplementation.h"
#include "circt/Dialect/Seq/SeqOpInterfaces.h"
#include "circt/Support/LLVM.h"
#include "mlir-c/IR.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/CodeGen/StackMaps.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TableGen/TableGenBackend.h"
#include <fstream>
#include <list>
#include <memory>
#include <mutex>
#include <ostream>
#include <set>
#include <stack>
#include <sys/types.h>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "circt/Dialect/HW/HWInstanceGraph.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "circt/Dialect/HW/HWPasses.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/FileSystem.h"

#include "mlir/Pass/AnalysisManager.h"

#include "mlir/IR/OperationSupport.h"
namespace circt {
namespace hw {
#define GEN_PASS_DEF_HWPOWERANALYZE
#include "circt/Dialect/HW/Passes.h.inc"
} // namespace hw
} // namespace circt

using namespace circt;
using namespace hw;

namespace {

struct HWPPAInfo {
    int fastDelay;
    int fastPower;
    int fastArea;
    int slowDelay;
    int slowPower;
    int slowArea;
};

struct HWPathNode {
    Operation * op;
    Value val;
    bool isTrueRoot;
    bool isTrueLeaf;
    HWPPAInfo ppaInfo;
};
using HWPathNodePtr = std::shared_ptr<HWPathNode>;

// using HWPath = std::list<HWPathNodePtr>;

struct Delay 
{
    int slow = 0;
    int fast = 0;
};
using DelayToNodes = std::unordered_multimap<HWPathNodePtr, Delay>;
struct HWNodeDelays {
    HWPathNodePtr node;

    // Value is delay

    DelayToNodes delaysIn;  // Delay from roots
    DelayToNodes delaysOut; // Delay to leaves
};
using HWNodeDelaysPtr = std::shared_ptr<HWNodeDelays>;

struct HWModulePPAModel {

    HWNodeDelaysPtr dfsPathForward(Value & next, int nextInd, HWNodeDelaysPtr & parent, int parInd);
    HWNodeDelaysPtr dfsPathBackward(Value & next, int nextInd, HWNodeDelaysPtr & child, int chiInd);
    // HWModulePPAModel(Operation *moduleOp, mlir::AnalysisManager &am);
    HWModulePPAModel(Operation *moduleOp);
    static HWModulePPAModel & getModel(Operation * moduleOp);

    static HWPPAInfo getPPAInfo(Operation * op);

    std::optional<Delay> getMaxDelayRoot2Leaf(Value & root, Value & leaf);
    std::optional<Delay> getMaxDelayLeafFromRoot(Value & root, Value & leaf);

    HWModulePPAModel & getModulesAnalysis(InstanceOp & op);


private:

    void traverseFromLeaf(Value & leaf, int ind);
    void traverseFromRoot(Value & root, int ind);

    std::unordered_map<llvm::hash_code, HWNodeDelaysPtr> rootBlocks;
    std::unordered_map<llvm::hash_code, HWNodeDelaysPtr> leafBlocks;
    std::unordered_map<llvm::hash_code, HWNodeDelaysPtr> blocks;
    std::unordered_map<Operation *, llvm::SmallVector<HWNodeDelaysPtr>> allOps;

    std::map<StringRef, HWModuleOp> siblings;

    std::map<StringRef, HWModulePPAModel> instances;
    std::map<StringRef, std::map<Value, Value>> instMaps;

    std::vector<Value> inpsList;
    std::vector<Value> outpsList;

    void getSibMods();
    bool processedSiblings = false;

    HWModuleOp analyzedOp;
    // mlir::AnalysisManager am;

    int foundLeaves = 0;

};

struct SignalInfo {
    std::string name;
    int width;
    uint64_t bitFlips = 0;
    std::string lastValue; // For multi-bit signals (binary string)
    uint64_t lastInt = 0;  // For multi-bit signals (integer)
    char lastScalar = 0;   // For scalar signals
    bool hasLastInt = false;
    double alpha = 0.5;
};
using SignalInfoPtr = std::shared_ptr<SignalInfo>;

struct VCDData{
    std::unordered_map<std::string, SignalInfoPtr> symbolToSignal;
    std::vector<std::string> hierarchyStack;
};

HWModulePPAModel & HWModulePPAModel::getModel(Operation * moduleOp)
{
    static std::unordered_map<Operation *, HWModulePPAModel> moduleModels;
    auto it = moduleModels.find(moduleOp);
    if (it == moduleModels.end())
    {
        it = moduleModels.insert({moduleOp, HWModulePPAModel(moduleOp)}).first;
    }
    return it->second;
}

struct HWPowerAnalyzePass
    : public circt::hw::impl::HWPowerAnalyzeBase<HWPowerAnalyzePass> {
    using Base::Base;

    void runOnOperation() override;
    void parseVCD();
    void walkOps();
    void calcSwitchingActivity();

    std::unordered_map<std::string, SignalInfoPtr> hierarchyToSignal;

private:

    const std::string dutBaseName = "svsimTestbench.dut";
    const std::string clockName = dutBaseName + ".clock";

    // VCD Parsing helpers
    bool parseVCDHeaderLine(VCDData & v, std::istringstream & iss);
    void parseVCDValueLine(VCDData & v, std::istringstream & iss);
    void parseVCDBitvectorLine(VCDData & v, std::istringstream & iss, std::string & token);
    void parseVCDBooleanLine(VCDData & v, std::string & token);
    void parseVCDVar(VCDData & v, std::istringstream & iss);

};
}

void HWModulePPAModel::getSibMods()
{
    llvm::errs() << "[PPA] Getting sibling HWModuleOps " << analyzedOp << '\n';
    Block * topBlock = analyzedOp->getBlock();
    for (Operation & module : topBlock->getOperations())
    {
        // narrow to the kind you care about (e.g. HWModuleOp)
        if (const HWModuleOp & hwMod = dyn_cast<HWModuleOp>(module)) {
            // request the analysis for that specific child operation
            // auto &model = getChildAnalysis<HWModulePPAModel>(hwMod);
            // llvm::errs() << "Name: " << mlir::SymbolTable::getSymbolName(hwMod) << '\n';
            auto name = mlir::SymbolTable::getSymbolName(hwMod).getValue();
            siblings.emplace(name, hwMod);
        }
    }
    processedSiblings = true;
}

HWPPAInfo HWModulePPAModel::getPPAInfo(Operation * op)
{
    if (!op) {
        return 
        {
            0,
            0,
            0,
            0,
            0,
            0
        };
    }
    return 
    {
        1,
        1,
        1,
        1,
        1,
        1
    };
}

std::optional<Delay> HWModulePPAModel::getMaxDelayRoot2Leaf(Value & root, Value & leaf)
{

    HWNodeDelaysPtr & rootBlk = rootBlocks.at(hash_value(root));
    HWPathNodePtr & leafBlk = leafBlocks.at(hash_value(leaf))->node;

    bool found = false;
    Delay maxDelay{0,0};
    for (auto & doIt : rootBlk->delaysOut) {
        HWPathNodePtr destNode = doIt.first;
        Delay d = doIt.second;
        if (destNode == leafBlk) {
            found = true;
            if (d.slow > maxDelay.slow)
                maxDelay.slow = d.slow;
            if (d.fast > maxDelay.fast)
                maxDelay.fast = d.fast;
        }
    }

    if (found)
        return maxDelay;
    return std::nullopt;
}

std::optional<Delay> HWModulePPAModel::getMaxDelayLeafFromRoot(Value & root, Value & leaf)
{
    HWNodeDelaysPtr & leafBlk = leafBlocks.at(hash_value(leaf));
    HWPathNodePtr & rootBlk = rootBlocks.at(hash_value(root))->node;

    bool found = false;
    Delay maxDelay{0,0};
    for (auto & diIt : leafBlk->delaysIn) {
        HWPathNodePtr srcNode = diIt.first;
        Delay d = diIt.second;
        if (srcNode == rootBlk) {
            found = true;
            if (d.slow > maxDelay.slow)
                maxDelay.slow = d.slow;
            if (d.fast > maxDelay.fast)
                maxDelay.fast = d.fast;
        }
    }

    if (found)
        return maxDelay;
    return std::nullopt;
}

HWModulePPAModel & HWModulePPAModel::getModulesAnalysis(InstanceOp & op)
{
    // Get analysis for every sibling HWModuleOp in the circuit:

    if (!processedSiblings) { getSibMods(); }

    auto model = instances.find(op.getModuleName());
    if (model == instances.end())
    {
        model = instances.emplace(
            op.getModuleName(),
            HWModulePPAModel::getModel(siblings.find(op.getModuleName())->second)
        ).first;

        // HWModuleOp mop = model->second.analyzedOp;
        // for (Value real : mop->getResults()) {
        
        // }

    }
    return model->second;
}


HWNodeDelaysPtr HWModulePPAModel::dfsPathBackward(Value & next, int nextInd, HWNodeDelaysPtr & child, int chiInd)
{
    // We traverse forward first, so all nodes should already be in blocks
    Value & curVal = next;
    int curInd = nextInd;
    Operation * defOp = curVal.getDefiningOp();
    if (isa<BlockArgument>(curVal)) {
        defOp = analyzedOp.getOperation();
    }
    auto blk = blocks.find(hash_value(curVal));
    
    if (blk == blocks.end())
    {
        HWPathNodePtr pn = std::make_shared<HWPathNode>();
        pn->op = defOp;
        pn->val = curVal;
        pn->isTrueRoot = false;
        pn->isTrueLeaf = false;
        pn->ppaInfo = getPPAInfo(defOp);

        HWNodeDelaysPtr newBlock = std::make_shared<HWNodeDelays>();
        newBlock->node = pn;

        blk = blocks.insert({hash_value(curVal), newBlock}).first;
    }
    HWNodeDelaysPtr & cur = blk->second;

    if (allOps.find(defOp) == allOps.end())
    {
        allOps.insert({defOp, llvm::SmallVector<HWNodeDelaysPtr>()});
    }
    allOps.at(defOp).push_back(cur);

    // If we haven't traversed this node before, traverse it
    if (!cur->delaysIn.size()) {

        // Check if this is a root node that we are dealing with
        if (dyn_cast<HWModuleOp>(defOp) || dyn_cast<seq::Clocked>(defOp) || dyn_cast<ConstantOp>(defOp)) {
            HWPathNodePtr pn = blk->second->node;
            cur->delaysIn.emplace(pn, Delay{pn->ppaInfo.slowDelay,pn->ppaInfo.fastDelay});
            pn->isTrueRoot = (dyn_cast<seq::Clocked>(defOp) || dyn_cast<ConstantOp>(defOp));
        }
        else if (InstanceOp instOp = dyn_cast<InstanceOp>(defOp))
        {
            HWModulePPAModel & model = getModulesAnalysis(instOp);
            Value & output = model.outpsList.at(curInd);
            HWNodeDelaysPtr paths = model.leafBlocks.find(hash_value(output))->second;
            for (auto & pi : paths->delaysIn) {
                if (!pi.first->isTrueRoot)
                {
                    continue;
                }
                Delay piD = pi.second;
                HWPathNodePtr pn = cur->node;
                pn->isTrueRoot = true;
                cur->delaysIn.emplace(pn, Delay{piD.slow, piD.fast});
            }

            for (uint i = 0; i < defOp->getNumOperands(); i++)
            {

                auto cp = model.getMaxDelayLeafFromRoot(model.inpsList.at(i), output);
                if (!cp.has_value()) {
                    // No path between root and leaf in the instance
                    continue;
                }
                Value operand = defOp->getOperand(i);
                HWNodeDelaysPtr par = dfsPathBackward(operand, i,  cur, curInd);
                for (auto & pi : par->delaysIn)
                {
                    int curSlow = cp->slow;
                    int curFast = cp->fast;
                    Delay piD = pi.second;
                    cur->delaysIn.emplace(pi.first, Delay{piD.slow + curSlow, piD.fast + curFast });
                }
            }
        }
        else // Otherwise, continue traversing
        {
            for (uint i = 0; i < defOp->getNumOperands(); i++)
            {
                Value operand = defOp->getOperand(i);
                HWNodeDelaysPtr par = dfsPathBackward(operand, i, cur, curInd);
                for (auto & pi : par->delaysIn)
                {
                    int curSlow = cur->node->ppaInfo.slowDelay;
                    int curFast = cur->node->ppaInfo.fastDelay;
                    Delay piD = pi.second;
                    cur->delaysIn.emplace(pi.first, Delay{piD.slow + curSlow, piD.fast + curFast });
                }
            }
        }
    }

    // if (child == nullptr)
    //     return cur;

    // // Give delayIn info to the current node's parent
    // int curSlow = cur->node->ppaInfo.slowDelay;
    // int curFast = cur->node->ppaInfo.fastDelay;
    // for (auto & pi : cur->delaysIn)
    // {
    //     Delay piD = pi.second;
    //     child->delaysIn.emplace(pi.first, Delay{piD.slow + curSlow, piD.fast + curFast });
    // }
    return cur;
}


HWNodeDelaysPtr HWModulePPAModel::dfsPathForward(Value & next, int nextInd, HWNodeDelaysPtr & parent, int parInd)
{
    // Rename
    Value & curVal = next;
    int curInd = nextInd;
    Operation * defOp = curVal.getDefiningOp();
    if (isa<BlockArgument>(curVal)) {
        defOp = analyzedOp.getOperation();
    }

    auto blk = blocks.find(hash_value(curVal));

    // If we haven't traversed this node before, construct a block add it to blocks
    if (blk == blocks.end())
    {
        HWPathNodePtr pn = std::make_shared<HWPathNode>();
        pn->op = defOp;
        pn->val = curVal;
        pn->isTrueRoot = false;
        pn->isTrueLeaf = false;
        pn->ppaInfo = getPPAInfo(defOp);

        HWNodeDelaysPtr newBlock = std::make_shared<HWNodeDelays>();
        newBlock->node = pn;

        blk = blocks.insert({hash_value(curVal), newBlock}).first;
    }
    HWNodeDelaysPtr & cur = blk->second;

    if (allOps.find(defOp) == allOps.end())
    {
        allOps.insert({defOp, llvm::SmallVector<HWNodeDelaysPtr>()});
    }
    allOps.at(defOp).push_back(cur);

    if (!cur->delaysOut.size())
    {
        // Now traverse the new block (the current block)
        for (auto & use : curVal.getUses()) {

            Operation * owner = use.getOwner();
            
            if (InstanceOp instOp = dyn_cast<InstanceOp>(owner))
            {
                HWModulePPAModel & model = getModulesAnalysis(instOp);
                
                Value & input = model.inpsList.at(use.getOperandNumber());
                HWNodeDelaysPtr paths = model.rootBlocks.find(hash_value(input))->second;
                for (auto & po : paths->delaysOut) {
                    if (!po.first->isTrueLeaf)
                    {
                        continue;
                    }
                    Delay poD = po.second;
                    HWPathNodePtr pn = cur->node;
                    pn->isTrueLeaf = true;
                    cur->delaysOut.emplace(pn, Delay{poD.slow, poD.fast});
                }

                for (uint i = 0; i < owner->getNumResults(); i++)
                {

                    auto cp = model.getMaxDelayRoot2Leaf(input, model.outpsList.at(i));
                    if (!cp.has_value()) {
                        // No path between root and leaf in the instance
                        continue;
                    }
                    OpResult res = owner->getResult(i);
                    dfsPathForward(res, i, cur, curInd);
                }
            }

            // Check if this use of cur is a leaf node (output or clocked op)
            else if (isa<OutputOp>(owner) || isa<seq::Clocked>(owner)) {
                HWPathNodePtr pn = blk->second->node;
                HWPPAInfo outpPPA = getPPAInfo(owner);
                cur->delaysOut.emplace(pn, Delay{outpPPA.slowDelay,outpPPA.fastDelay});
                pn->isTrueLeaf = dyn_cast<seq::Clocked>(owner);
                foundLeaves++;

                if (allOps.find(owner) == allOps.end())
                {
                    allOps.insert({owner, llvm::SmallVector<HWNodeDelaysPtr>()});
                }

            } else
            { // Otherwise, continue traversing
                for (uint i = 0; i < owner->getNumResults(); i++)
                {

                    OpResult res = owner->getResult(i);
                    dfsPathForward(res, i, cur, curInd);
                }
            }
        }
    }

    if (parent == nullptr)
        return cur;

    // Give delayOut info to the parent about current's children

    // IF cur->node->op is an instance, we need to adjust the delays here.
    // IF THERE IS NO PATH BETWEEN A ROOT AND LEAF WE SHOULD NOT INCLUDE IT IN THE PATHS
    int curSlow;
    int curFast;
    if (InstanceOp instOp = dyn_cast<InstanceOp>(cur->node->op))
    {
        HWModulePPAModel & model = getModulesAnalysis(instOp);

        auto cp = model.getMaxDelayRoot2Leaf(model.inpsList.at(parInd), model.outpsList.at(nextInd));
        // model.dfsPathForward(Value &next, HWNodeDelaysPtr &parent)

        if (!cp.has_value()) {
            // No path between root and leaf in the instance
            return cur; 
        }

        curSlow = cp->slow; // CUR SLOW IS THE PATH FROM THE INPUT (PARENT) TO THE OUTPUT (CURRENT)
        curFast = cp->fast; // CUR FAST IS THE PATH FROM THE INPUT (PARENT) TO THE OUTPUT (CURRENT)
    } else
    {
        curSlow = cur->node->ppaInfo.slowDelay;
        curFast = cur->node->ppaInfo.fastDelay;
    }

    for (auto & po : cur->delaysOut) {
        Delay poD = po.second;
        parent->delaysOut.emplace(po.first, Delay{poD.slow + curSlow, poD.fast + curFast });
    }

    return cur;
}

// HWNodeDelaysPtr HWModulePPAModel::dfsPathForward(Operation * next, int nextInd, HWNodeDelaysPtr & parent, int parInd)
// {
//     // Rename

//     Operation * curOp = next;
//     int curInd = nextInd;

//     auto blk = blocks.find(curOp);

//     // If we haven't traversed this node before, construct a block add it to blocks
//     if (blk == blocks.end())
//     {
//         HWPathNodePtr pn = std::make_shared<HWPathNode>();
//         pn->op = curOp;
//         pn->isTrueRoot = false;
//         pn->isTrueLeaf = false;
//         pn->ppaInfo = getPPAInfo(curOp);

//         HWNodeDelaysPtr newBlock = std::make_shared<HWNodeDelays>();
//         newBlock->node = pn;

//         blk = blocks.insert({curOp, newBlock}).first;
//     }
//     HWNodeDelaysPtr & cur = blk->second;

//     if (isa<OutputOp>(curOp) || isa<seq::Clocked>(curOp)) {
//         HWPathNodePtr pn = blk->second->node;
//         HWPPAInfo outpPPA = getPPAInfo(curOp);
//         cur->delaysOut.emplace(pn, Delay{outpPPA.slowDelay,outpPPA.fastDelay});
//         pn->isTrueLeaf = dyn_cast<seq::Clocked>(curOp);
//         foundLeaves++;
//     }

//     if (!cur->delaysOut.size())
//     {
//         for (auto result : curOp->getResults()) 
//         {

//             for (auto & use : result.getUses())
//             {
//                 Operation * owner = use.getOwner();
            
//                 if (InstanceOp instOp = dyn_cast<InstanceOp>(owner))
//                 {
//                     HWModulePPAModel & model = getModulesAnalysis(instOp);
                    
//                     Value & input = model.inpsList.at(use.getOperandNumber());
//                     HWNodeDelaysPtr paths = model.rootBlocks.find(hash_value(input))->second;
//                     for (auto & po : paths->delaysOut) {
//                         if (!po.first->isTrueLeaf)
//                         {
//                             continue;
//                         }
//                         Delay poD = po.second;
//                         HWPathNodePtr pn = cur->node;
//                         pn->isTrueLeaf = true;
//                         cur->delaysOut.emplace(pn, Delay{poD.slow, poD.fast});
//                     }

//                     for (uint i = 0; i < owner->getNumResults(); i++)
//                     {

//                         auto cp = model.getMaxDelayRoot2Leaf(input, model.outpsList.at(i));
//                         if (!cp.has_value()) {
//                             // No path between root and leaf in the instance
//                             continue;
//                         }
//                         OpResult res = owner->getResult(i);
//                         dfsPathForward(res, i, cur, curInd);
//                     }
//                 } else
//                 { // Otherwise, continue traversing
//                     dfsPathForward(owner, use.getOperandNumber(), cur, curInd);
//                 }
            
            
//             }

//         }
//     }
//     if (parent == nullptr)
//         return cur;

//     // Give delayOut info to the parent about current's children

//     // IF cur->node->op is an instance, we need to adjust the delays here.
//     // IF THERE IS NO PATH BETWEEN A ROOT AND LEAF WE SHOULD NOT INCLUDE IT IN THE PATHS
//     int curSlow;
//     int curFast;
//     if (InstanceOp instOp = dyn_cast<InstanceOp>(cur->node->op))
//     {
//         HWModulePPAModel & model = getModulesAnalysis(instOp);

//         auto cp = model.getMaxDelayRoot2Leaf(model.inpsList.at(parInd), model.outpsList.at(nextInd));
//         // model.dfsPathForward(Value &next, HWNodeDelaysPtr &parent)

//         if (!cp.has_value()) {
//             // No path between root and leaf in the instance
//             return cur; 
//         }

//         curSlow = cp->slow; // CUR SLOW IS THE PATH FROM THE INPUT (PARENT) TO THE OUTPUT (CURRENT)
//         curFast = cp->fast; // CUR FAST IS THE PATH FROM THE INPUT (PARENT) TO THE OUTPUT (CURRENT)
//     } else
//     {
//         curSlow = cur->node->ppaInfo.slowDelay;
//         curFast = cur->node->ppaInfo.fastDelay;
//     }

//     for (auto & po : cur->delaysOut) {
//         Delay poD = po.second;
//         parent->delaysOut.emplace(po.first, Delay{poD.slow + curSlow, poD.fast + curFast });
//     }

//     return cur;
// }

void HWModulePPAModel::traverseFromLeaf(Value & leaf, int ind) {
    if (leafBlocks.find(hash_value(leaf)) != leafBlocks.end())
    {
        return;
    }

    HWNodeDelaysPtr nullPtr = nullptr;
    HWNodeDelaysPtr leafBlock = dfsPathBackward(leaf, 0, nullPtr, -1);
    leafBlocks.emplace(hash_value(leaf), leafBlock);
};

void HWModulePPAModel::traverseFromRoot(Value & root, int ind) {
    // I don't think this should ever evaluate to true.
    if (rootBlocks.find(hash_value(root)) != rootBlocks.end())
    {
        return;
    }

    HWNodeDelaysPtr nullPtr = nullptr;
    HWNodeDelaysPtr rootBlock = dfsPathForward(root, 0, nullPtr, -1);
    rootBlocks.insert({hash_value(root), rootBlock});
};

std::mutex debugWrLock;

HWModulePPAModel::HWModulePPAModel(Operation * moduleOp)
{

    HWModuleOp mop = dyn_cast<HWModuleOp>(moduleOp);
    if (!mop) {
        return;
    }
    llvm::errs() << "MOP ADDRESS" << mop.getOperation() << '\n';
    llvm::errs() << "PreMOP Address" << moduleOp << '\n';
    analyzedOp = mop;

    for (size_t i = 0; i < mop.getNumInputPorts(); i++)
    {
        BlockArgument res = mop.getArgumentForInput(i);
        inpsList.push_back(static_cast<Value>(res));
        traverseFromRoot(static_cast<Value&>(res), i);
    }

    mop->walk([&](Operation * op) {

        seq::Clocked clockedOp = dyn_cast<seq::Clocked>(op);
        if (clockedOp)
        {
            for (uint i = 0; i < op->getNumResults(); i++)
            {
                OpResult result = op->getResult(i);
                traverseFromRoot(static_cast<Value&>(result), i);
            }
            for (uint i = 0; i < op->getNumOperands(); i++)
            {
                Value operand = op->getOperand(i);
                traverseFromLeaf(operand, i);
            }
            return;
        }
        ConstantOp constOp = dyn_cast<ConstantOp>(op);
        if (constOp)
        {
            for (uint i = 0; i < op->getNumResults(); i++)
            {
                OpResult result = op->getResult(i);
                traverseFromRoot(static_cast<Value&>(result), i);
            }
            return;
        }
        OutputOp outputOp = dyn_cast<OutputOp>(op);
        if (outputOp)
        {
            for (uint i = 0; i < op->getNumOperands(); i++)
            {
                Value operand = op->getOperand(i);
                outpsList.push_back(operand);
                traverseFromLeaf(operand, i);
            }
        }
    });

    
    llvm::errs() << "[PPA] Module " << mop.getName() << '\n';
    llvm::errs() << "[PPA] have " << blocks.size() << " blocks\n";
    llvm::errs() << "[PPA] have " << foundLeaves << " found leaves\n";
    llvm::errs() << "[PPA] have " << leafBlocks.size() << " blocks leaves\n";
    for (auto & lb : leafBlocks)
    {
        llvm::errs() << lb.second->node->val << '\n';
    }
    llvm::errs() << "[PPA] have " << rootBlocks.size() << " blocks roots\n";
    for (auto & rb : rootBlocks)
    {
        llvm::errs() << rb.second->node->val << '\n';
    }

    for (auto & rb : rootBlocks)
    {
        for (auto & lb : leafBlocks)
        {
            std::optional<Delay> d = getMaxDelayRoot2Leaf(rb.second->node->val, lb.second->node->val);
            if (d)
            {
                llvm::errs() << "[PPA] Max delay from root " << rb.second->node->val << " to leaf " << lb.second->node->val << " is slow: " << d->slow << ", fast: " << d->fast << '\n';
            }
            d = getMaxDelayLeafFromRoot(rb.second->node->val, lb.second->node->val);
            if (d)
            {
                llvm::errs() << "[PPA] Max delay to leaf " << lb.second->node->val << " from root " << rb.second->node->val << " is slow: " << d->slow << ", fast: " << d->fast << '\n';
                llvm::errs() << "\n";
            }
        }
    }

    for (auto & block : blocks) {

        llvm::errs() << "[PPA] Node " << block.second->node->val << '\n';
        // if (block.second->node->op)
        // {
        //     block.second->node->op->dumpPretty();
        // }
        // else {
        //     llvm::errs() << "mod inp";
        // }

        for (auto & pi : block.second->delaysIn) {
            llvm::errs() << "\n    delayIn from " << pi.first->val;
            // if (pi.first->op) { 
            //     pi.first->op->dumpPretty();
            // } else {
            //     llvm::errs() << "mod inp";
            // }
            llvm::errs() << " is: " << pi.second.fast << '\n';
        }
        for (auto & po : block.second->delaysOut) {
            llvm::errs() << "\n    delayOut to " << po.first->val;
            // if (po.first->op) {
            //     po.first->op->dumpPretty();
            // } else {
            //     llvm::errs() << "mod out";
            // }
            llvm::errs() << " is: " << po.second.fast << '\n';
        }
    }
    llvm::errs() << "\n\n\n";
    
    // debugWrLock.unlock();


    // (2) run DFS a second time to annotate all nodes with the inputs they depend on
    // (3) Go through all nodes in topological order, assigning a fast and slow hw block to each op, and calculating slow and fast path lengths from roots to leaves.


    // Clocked objects have results which are roots and inputs which are leaves



}

void HWPowerAnalyzePass::parseVCDVar(VCDData & v, std::istringstream & iss)
{
    std::unordered_map<std::string, SignalInfoPtr> & symbolToSignal = v.symbolToSignal;
    std::vector<std::string> & hierarchyStack = v.hierarchyStack;

    std::string varType;
    int width;
    std::string symbol, name;
    iss >> varType >> width >> symbol >> name;
    // Handle bus names with [xx:yy]
    // std::string rest;
    // std::getline(iss, rest);
    // name += rest;
    std::string fullName;
    for (const auto &h : hierarchyStack) {
        if (!fullName.empty())
            fullName += ".";
        fullName += h;
    }
    if (!fullName.empty())
        fullName += ".";
    fullName += name;

    auto it = symbolToSignal.find(symbol);
    SignalInfoPtr ptr;
    if (it == symbolToSignal.end()) {
        ptr = std::make_shared<SignalInfo>();
        ptr->name = name;
        ptr->width = width;
        symbolToSignal[symbol] = ptr;
    } else
    {
        ptr = it->second;
    }
    hierarchyToSignal[fullName] = ptr;
}

bool HWPowerAnalyzePass::parseVCDHeaderLine(VCDData & v, std::istringstream & iss)
{
    std::string token;
    iss >> token;

    std::vector<std::string> & hierarchyStack = v.hierarchyStack;

    if (token == "$scope") {
        std::string type, name;
        iss >> type >> name;
        hierarchyStack.push_back(name);
    } else if (token == "$upscope") {
        if (!hierarchyStack.empty())
            hierarchyStack.pop_back();
    } else if (token == "$var") {
        parseVCDVar(v, iss);
    } else if (token == "$enddefinitions") {
        return false;
    }
    return true;
}

void HWPowerAnalyzePass::parseVCDBooleanLine(VCDData & v, std::string & token)
{
    std::unordered_map<std::string, SignalInfoPtr> & symbolToSignal = v.symbolToSignal;

    char value = token[0];
    if (value == 'x' || value == 'z')
        return;
    std::string symbol = token.substr(1);
    auto it = symbolToSignal.find(symbol);
    if (it != symbolToSignal.end()) {
        SignalInfo &sig = *it->second;
        if (sig.lastScalar && value != sig.lastScalar)
            sig.bitFlips++;
        sig.lastScalar = value;
    }
}

void HWPowerAnalyzePass::parseVCDBitvectorLine(VCDData & v, std::istringstream & iss, std::string & token)
{
    std::unordered_map<std::string, SignalInfoPtr> & symbolToSignal = v.symbolToSignal;

    std::string value = token.substr(1);
    std::string symbol;
    iss >> symbol;
    auto it = symbolToSignal.find(symbol);
    if (it != symbolToSignal.end()) {
        SignalInfo &sig = *it->second;
        // Convert binary string to integer
        uint64_t currInt = 0;
        if (!value.empty() &&
            value.find_first_not_of("01") == std::string::npos) {
            currInt = std::stoull(value, nullptr, 2);
        } else {
            llvm::errs()
                << "[VCD] Warning: Non-binary value '" << value
                << "' for signal " << symbol << "\n";
        }
        if (sig.hasLastInt) {
            uint64_t xorVal = currInt ^ sig.lastInt;
#if defined(__GNUC__) || defined(__clang__)
            sig.bitFlips += __builtin_popcountll(xorVal);
#else
            // Portable popcount fallback
            uint64_t x = xorVal;
            int count = 0;
            while (x) {
                count += x & 1;
                x >>= 1;
            }
            sig.bitFlips += count;
#endif
        }
        sig.lastInt = currInt;
        sig.hasLastInt = true;
        sig.lastValue = value;
    }
}

void HWPowerAnalyzePass::parseVCD()
{
    if (this->powerAnalyzeVcd.empty())
        return;

    std::ifstream vcd(this->powerAnalyzeVcd);
    if (!vcd) {
        llvm::errs() << "[PowerAnalyze] Could not open VCD file: "
                     << this->powerAnalyzeVcd << "\n";
        return;
    }

    std::string line;

    VCDData v = {};

    bool more = true;
    while (std::getline(vcd, line) && more) {
        std::istringstream iss(line);
        more = parseVCDHeaderLine(v, iss);
    }
    while (std::getline(vcd, line)) {
        std::istringstream iss(line);
        std::string token;
        iss >> token;

        if (token[0] == 'b') {
            parseVCDBitvectorLine(v, iss, token);
        } else if (token.size() > 1 || isalnum(token[0])) {
            parseVCDBooleanLine(v, token);
        }
    }

    // Print bit flip summary
    for (const auto &kv : hierarchyToSignal) {
        llvm::errs() << "[VCD] " << kv.first
                     << " bit flips: " << kv.second->bitFlips << "\n";
    }
}

void HWPowerAnalyzePass::calcSwitchingActivity()
{
    int numCycles = hierarchyToSignal.at(clockName)->bitFlips;

    // For each signal in each module, calculate switching activity
    for (auto &kv : hierarchyToSignal) {
        SignalInfoPtr & sig = kv.second;
        double alpha = static_cast<double>(sig->bitFlips) /
                       (sig->width * numCycles);
        llvm::errs() << "[PowerAnalyze] " << kv.first
                     << " switching activity (alpha): " << alpha << "\n";
        sig->alpha = alpha;
    }
}

void HWPowerAnalyzePass::walkOps()
{

    auto circuitOp = getOperation();
    llvm::errs() << "[PowerAnalyze] Analyzing circuit: " << circuitOp.getOperationName() << "\n";



    circuitOp->walk([&](Operation *op) {
        llvm::errs() << "[PowerAnalyze] Found operation: " << op->getName() << "\n";
    
        op->getParentOp();
    });

    

    // circuitOp->walk(FnT &&callback);
}

HWPowerAnalyzePass * thePass;


void HWPowerAnalyzePass::runOnOperation() 
{
    thePass = this;
    debugWrLock.lock();

    llvm::errs() << "[PowerAnalyze] Running power analysis pass\n";
    llvm::errs().flush();

    mlir::ModuleOp topOp = getOperation();
    topOp->walk([&](HWModuleOp modOp) {

        HWModulePPAModel & a = HWModulePPAModel::getModel(modOp);

    });

    // Block * topBlock = getOperation()->getBlock();
    // for (Operation & module : topBlock->getOperations())
    // {
    //     // narrow to the kind you care about (e.g. HWModuleOp)
    //     if (const HWModuleOp & hwMod = dyn_cast<HWModuleOp>(module)) {
    //         // request the analysis for that specific child operation
    //         // auto &model = getChildAnalysis<HWModulePPAModel>(hwMod);
    //         llvm::errs() << "Name: " << mlir::SymbolTable::getSymbolName(hwMod).getValue() << '\n';
    //     }
    // }
    
    debugWrLock.unlock();

    // getAnalysis<typename AnalysisT>()
    // getAnalysis<HWModulePPAModel>();
    
    // getChildAnalysis();

    // getAnalysis<HWModulePPAModel>();

    return;

    walkOps();

    // Print the VCD file option if provided
    if (!powerAnalyzeVcd.empty()) {
        llvm::errs() << "[PowerAnalyze] Using VCD file for switching activity: "
                     << powerAnalyzeVcd << "\n";
    }
    // Let's parse this VCD :sunglasses:
    if (!powerAnalyzeVcd.empty()) {
        parseVCD();
    }
    calcSwitchingActivity();

    // llvm::errs() << "###" << i++ << ": " << module.getName() << "\n";
    // for (auto name : module.getPorts()) {
    //     llvm::errs() << name.getName() << '\n';
    // }

    // First, for each module in the project, create a tech-agnostic power
    // and area model, where the power is parameterized by switching activity

    // Working backwards
    // In the end I want power & area estimates

    // For signal in each module, if using switching I need average bit flip
    // rates for each wire This'll take in a vcd. The VCD will be parsed in 1
    // shot.

    // A hash mapping from signal to # of flipped bits throughout the entire sim
    // Alpha will then be # of flipped bits / (bitwidth * num of cycles).

    // We should start there, then, if we need to, additionally model clock
    // power consumption. Also fanout for shallow modules (doesn't behoove us
    // to include fanout for a multiplier probably, but for bitwise ops the
    // fanout may be the primary consideration.)

    // https://www.chisel-lang.org/docs/explanations/testing Chisel sim for vcd






}



// SOME OLD DEBUG PRINTS

// debugWrLock.lock();

// llvm::errs() << "[PPA] Module " << mop.getName() << '\n';
// llvm::errs() << "[PPA] Found " << roots.size() << " roots and " << leaves.size() << " leaves\n";
// llvm::errs() << "[PPA] Roots:\n";
// for (Value v : roots)
// {
//     llvm::errs() << "  Value: " << v << '\n';
//     for (auto &use : v.getUses())
//     {
//         llvm::errs() << "    use: " << *use.getOwner() << '\n';
//     }
//     llvm::errs() << '\n';
//     // if (Operation *def = v.getDefiningOp())
//     // {
//     //     llvm::errs() << "  def: " << *def << '\n';
//     // }
//     // else if (auto ba = dyn_cast<BlockArgument>(v))
//     // {
//     //     llvm::errs() << "  arg: " << ba << '\n';
//     // }
// }
// llvm::errs() << "[PPA] Leaves:\n";
// for (Value v : leaves)
// {
//     llvm::errs() << "  Value: " << v << '\n';
//     for (auto &use : v.getUses())
//     {
//         llvm::errs() << "    use: " << *use.getOwner() << '\n';
//     }
//     llvm::errs() << '\n';
//     // if (Operation *def = v.getDefiningOp())
//     // {
//     //     llvm::errs() << "  def: " << *def << '\n';
//     // }
//     // else if (auto ba = dyn_cast<BlockArgument>(v))
//     // {
//     //     llvm::errs() << "  arg: " << ba << '\n';
//     // }
// }
// llvm::errs() << "\n\n";

// debugWrLock.unlock();
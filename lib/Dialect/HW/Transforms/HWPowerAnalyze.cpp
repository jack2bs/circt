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
#include <stack>
#include <unordered_map>
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

    HWNodeDelaysPtr dfsPathForward(Value & next, HWNodeDelaysPtr & parent);
    HWNodeDelaysPtr dfsPathBackward(Value & next, HWNodeDelaysPtr & child);
    HWModulePPAModel(Operation *moduleOp, mlir::AnalysisManager &am);

    static HWPPAInfo getPPAInfo(Operation * op);

private:

    void traverseFromLeaf(Value & leaf);
    void traverseFromRoot(Value & root);

    std::unordered_map<llvm::hash_code, HWNodeDelaysPtr> rootBlocks;
    std::unordered_map<llvm::hash_code, HWNodeDelaysPtr> leafBlocks;
    std::unordered_map<llvm::hash_code, HWNodeDelaysPtr> blocks;
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

HWPPAInfo HWModulePPAModel::getPPAInfo(Operation * op)
{
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

HWNodeDelaysPtr HWModulePPAModel::dfsPathBackward(Value & next, HWNodeDelaysPtr & child)
{
    // We traverse forward first, so all nodes should already be in blocks
    Value & curVal = next;
    Operation * defOp = curVal.getDefiningOp();
    auto blk = blocks.find(hash_value(curVal));
    
    if (blk == blocks.end())
    {
        HWPathNodePtr pn = std::make_shared<HWPathNode>();
        pn->op = defOp;
        pn->val = curVal;
        pn->isTrueRoot = false;
        pn->isTrueLeaf = false;
        pn->ppaInfo = defOp ? getPPAInfo(defOp) : HWPPAInfo{0};

        HWNodeDelaysPtr newBlock = std::make_shared<HWNodeDelays>();
        newBlock->node = pn;

        blk = blocks.insert({hash_value(curVal), newBlock}).first;
    }

    HWNodeDelaysPtr & cur = blk->second;

    // If we haven't traversed this node before, traverse it
    if (!cur->delaysIn.size()) {

        // Check if this is a root node that we are dealing with
        if (!defOp || dyn_cast<seq::Clocked>(defOp) || dyn_cast<ConstantOp>(defOp)) {
            HWPathNodePtr pn = blk->second->node;
            cur->delaysIn.emplace(pn, Delay{pn->ppaInfo.slowDelay,pn->ppaInfo.fastDelay});
            pn->isTrueRoot = defOp && (dyn_cast<seq::Clocked>(defOp) || dyn_cast<ConstantOp>(defOp));
        }
        else // Otherwise, continue traversing
        {
            for (auto operand : defOp->getOperands())
            {
                dfsPathBackward(operand, cur);
            }
        }
    }

    if (child == nullptr)
        return cur;

    // Give delayIn info to the current node's parent
    int curSlow = cur->node->ppaInfo.slowDelay;
    int curFast = cur->node->ppaInfo.fastDelay;
    for (auto & pi : cur->delaysIn)
    {
        Delay piD = pi.second;
        child->delaysIn.emplace(pi.first, Delay{piD.slow + curSlow, piD.fast + curFast });
    }
    return cur;
}

HWNodeDelaysPtr HWModulePPAModel::dfsPathForward(Value & next, HWNodeDelaysPtr & parent)
{
    // Rename
    Value & curVal = next;
    Operation * defOp = curVal.getDefiningOp();
    auto blk = blocks.find(hash_value(curVal));

    // If we haven't traversed this node before, construct a block add it to blocks
    if (blk == blocks.end())
    {
        HWPathNodePtr pn = std::make_shared<HWPathNode>();
        pn->op = defOp;
        pn->val = curVal;
        pn->isTrueRoot = false;
        pn->isTrueLeaf = false;
        pn->ppaInfo = defOp ? getPPAInfo(defOp) : HWPPAInfo{0};

        HWNodeDelaysPtr newBlock = std::make_shared<HWNodeDelays>();
        newBlock->node = pn;

        blk = blocks.insert({hash_value(curVal), newBlock}).first;
    }

    HWNodeDelaysPtr & cur = blk->second;
    if (!cur->delaysOut.size())
    {
        // Now traverse the new block (the current block)
        for (auto & use : curVal.getUses()) {

            Operation * owner = use.getOwner();

            // Check if this use of cur is a leaf node (output or clocked op)
            if (isa<OutputOp>(owner) || dyn_cast<seq::Clocked>(owner)) {
                HWPathNodePtr pn = blk->second->node;
                cur->delaysOut.emplace(pn, Delay{pn->ppaInfo.slowDelay,pn->ppaInfo.fastDelay});
                pn->isTrueLeaf = dyn_cast<seq::Clocked>(owner);
                foundLeaves++;
            } else { // Otherwise, continue traversing
                for (auto res : owner->getResults())
                {
                    dfsPathForward(res, cur);
                }
            }
        }
    }

    if (parent == nullptr)
        return cur;

    // Give delayOut info to the parent about current's children
    int curSlow = cur->node->ppaInfo.slowDelay;
    int curFast = cur->node->ppaInfo.fastDelay;
    for (auto & po : cur->delaysOut) {
        Delay poD = po.second;
        parent->delaysOut.emplace(po.first, Delay{poD.slow + curSlow, poD.fast + curFast });
    }

    return cur;
}

void HWModulePPAModel::traverseFromLeaf(Value & leaf) {
    if (leafBlocks.find(hash_value(leaf)) != leafBlocks.end())
    {
        return;
    }

    HWNodeDelaysPtr nullPtr = nullptr;
    HWNodeDelaysPtr leafBlock = dfsPathBackward(leaf, nullPtr);
    leafBlocks.emplace(hash_value(leaf), leafBlock);
};

void HWModulePPAModel::traverseFromRoot(Value & root) {
    // I don't think this should ever evaluate to true.
    if (rootBlocks.find(hash_value(root)) != rootBlocks.end())
    {
        return;
    }

    HWNodeDelaysPtr nullPtr = nullptr;
    HWNodeDelaysPtr rootBlock = dfsPathForward(root, nullPtr);
    rootBlocks.insert({hash_value(root), rootBlock});
};

std::mutex debugWrLock;

HWModulePPAModel::HWModulePPAModel(Operation * moduleOp, mlir::AnalysisManager &am)
{

    HWModuleOp mop = dyn_cast<HWModuleOp>(moduleOp);
    if (!mop) {
        return;
    }

    for (size_t i = 0; i < mop.getNumInputPorts(); i++)
    {
        BlockArgument res = mop.getArgumentForInput(i);
        traverseFromRoot(static_cast<Value&>(res));
    }

    mop->walk([&](Operation * op) {

        seq::Clocked clockedOp = dyn_cast<seq::Clocked>(op);
        if (clockedOp)
        {
            for (OpResult result : op->getResults())
            {
                traverseFromRoot(static_cast<Value&>(result));
            }
            for (Value operand : op->getOperands())
            {
                traverseFromLeaf(operand);
            }
            return;
        }
        ConstantOp constOp = dyn_cast<ConstantOp>(op);
        if (constOp)
        {
            for (OpResult result : op->getResults())
            {
                traverseFromRoot(static_cast<Value&>(result));
            }
            return;
        }
        OutputOp outputOp = dyn_cast<OutputOp>(op);
        if (outputOp)
        {
            for (Value operand : outputOp->getOperands())
            {
                traverseFromLeaf(operand);
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

    debugWrLock.unlock();


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


void HWPowerAnalyzePass::runOnOperation() 
{
    debugWrLock.lock();

    llvm::errs() << "[PowerAnalyze] Running power analysis pass\n";
    getAnalysis<HWModulePPAModel>();
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
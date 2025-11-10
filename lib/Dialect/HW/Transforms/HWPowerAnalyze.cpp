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

#include "circt/Dialect/HW/HWOpInterfaces.h"
#include "circt/Dialect/HW/PortImplementation.h"
#include "circt/Dialect/Seq/SeqOpInterfaces.h"
#include "circt/Support/LLVM.h"
#include "mlir-c/IR.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/CodeGen/StackMaps.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/Timer.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TableGen/TableGenBackend.h"
#include <algorithm>
#include <cstddef>
#include <cstdint>
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

#include <thread>
#include <condition_variable>

#include "circt/Dialect/HW/HWInstanceGraph.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWPasses.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/FileSystem.h"

#include "mlir/Pass/AnalysisManager.h"

#include "mlir/IR/OperationSupport.h"

#include "llvm/Support/JSON.h"
#include <fstream>
#include <cstdlib>

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
    double delay = 0.0;
    double area = 0.0;
    double lkgpwr = 0.0;
    double dynengy = 0.0;
    int in1Width = 0;
    int auxWidth = 0;

    bool operator>(HWPPAInfo const & other) const
    {
        if (this->delay == other.delay)
        {
            return this->area > other.area;
        }
        return this->delay > other.delay;
    }
};
using HWPPASet = std::set<HWPPAInfo, std::greater<>>;
using HWPPASetPtr = std::shared_ptr<HWPPASet>;

class HWPPAGroup {
public:
    void addInfo(HWPPAInfo & ppaInfo)
    {
        for (auto i = m_infoSets.begin(); i != m_infoSets.end(); i++) {
            auto & auxWidthVec = *i;
            int existingInWid = auxWidthVec.at(0)->begin()->in1Width;
            if (existingInWid < ppaInfo.in1Width)
            {
                continue;
            }
            if (existingInWid > ppaInfo.in1Width)
            {
                auto newSetVec = m_infoSets.insert(i, std::vector<HWPPASetPtr>());
                auto newSet = newSetVec->emplace_back(std::make_shared<HWPPASet>());
                newSet->insert(ppaInfo);
                return;
            }
            for (auto j = auxWidthVec.begin(); j != auxWidthVec.end(); j++)
            {
                auto & set = *j;
                if (set->begin()->auxWidth < ppaInfo.auxWidth)
                {
                    continue;
                }
                if (set->begin()->auxWidth > ppaInfo.auxWidth)
                {
                    auto newSet = auxWidthVec.emplace_back(std::make_shared<HWPPASet>());
                    newSet->insert(ppaInfo);
                    return;
                }
                set->insert(ppaInfo);
                return;
            }
            auto newSet = auxWidthVec.emplace_back(std::make_shared<HWPPASet>());
            newSet->insert(ppaInfo);
            return;
        }
        auto & newSetVec = m_infoSets.emplace_back();
        auto & newSet = newSetVec.emplace_back(std::make_shared<HWPPASet>());
        newSet->insert(ppaInfo);
    }

    HWPPASetPtr getBestMatch(int in1Width, int auxWidth) const
    {
        for (auto & auxWidthVec : m_infoSets) {
            int existingInWid = auxWidthVec.at(0)->begin()->in1Width;
            if (existingInWid < in1Width)
            {
                continue;
            }
            for (auto & set : auxWidthVec)
            {
                if (set->begin()->auxWidth < auxWidth)
                {
                    continue;
                }
                return set;
            }
        }
        llvm::errs() << "[PPAAnalyze] No profile can handle the inputs to getBestMatch which are: " << in1Width << " and " << auxWidth << '\n';
        llvm::errs() << "  For group: " << m_infoSets.begin()->begin()->get()->begin()->delay << '\n';
        return nullptr;
    }

    std::vector<std::vector<HWPPASetPtr>> m_infoSets;
};
using HWPPAGroupPtr = std::shared_ptr<HWPPAGroup>;

struct HWPPAData {
    std::unordered_map<std::string, HWPPAGroup> data;
};

struct HWPathNode {
    Operation * op;
    Value val;
    // bool isTrueRoot;
    // bool isTrueLeaf;
    HWPPASetPtr ppaInfo;
};
using HWPathNodePtr = std::shared_ptr<HWPathNode>;

// using HWPath = std::list<HWPathNodePtr>;

// struct Delay 
// {
//     int slow = 0;
//     int fast = 0;
// };
struct DelayPath
{
    HWPathNodePtr node;
    double delay;
    bool isTrue;
    int fromInd;
    int toInd;
};

using DelaysFromRoot = std::vector<DelayPath>;
using DelaysToLeaf = std::vector<DelayPath>;
struct HWNodeDelays {
    HWPathNodePtr node;

    llvm::SmallVector<int> forwardIndices;
    llvm::SmallVector<int> backwardIndices;

    DelaysFromRoot delaysIn;  // Delay from roots
    DelaysToLeaf delaysOut; // Delay to leaves
};
using HWNodeDelaysPtr = std::shared_ptr<HWNodeDelays>;

struct HWModulePPAModel {

    HWNodeDelaysPtr dfsPathForward(Operation * next, int nextInd, HWNodeDelaysPtr & parent, int parInd);
    HWNodeDelaysPtr dfsPathBackward(Operation * next, int nextInd, HWNodeDelaysPtr & child, int chiInd);
    // HWModulePPAModel(Operation *moduleOp, mlir::AnalysisManager &am);
    HWModulePPAModel(Operation *moduleOp, const HWPPAData * ppaDataPtr);
    static HWModulePPAModel & getModel(Operation * moduleOp, const HWPPAData * ppaData);

    static bool isZeroCostOp(Operation * op);
    struct OperandInfo 
    {
        bool isConstant;
        int width;
    };
    static OperandInfo getOperandInfo(Value operand);
    HWPPASetPtr getPPAInfo(Operation * op);

    // std::optional<Delay> getMaxDelayRoot2Leaf(Value & root, Value & leaf);
    // std::optional<Delay> getMaxDelayLeafFromRoot(Value & root, Value & leaf);

    HWModulePPAModel & getModulesAnalysis(InstanceOp & op);


private:

    void traverseFromLeaf(Value & leaf, int ind);
    void traverseFromRoot(Value & root, int ind);
    // void traverseFromRoot(Operation * root);

    std::unordered_map<llvm::hash_code, DelaysToLeaf> rootBlocks;
    std::unordered_map<llvm::hash_code, DelaysFromRoot> leafBlocks;
    // std::unordered_map<llvm::hash_code, HWNodeDelaysPtr> blocks;
    std::unordered_map<Operation *, HWNodeDelaysPtr> blocks;
    std::unordered_set<Operation *> forwardVisited;
    std::unordered_set<Operation *> backwardVisited;
    // std::unordered_map<Operation *, llvm::SmallVector<HWNodeDelaysPtr>> allOps;

    std::map<StringRef, HWModuleOp> siblings;

    std::map<StringRef, HWModulePPAModel> instances;
    std::map<StringRef, std::map<Value, Value>> instMaps;

    std::vector<Value> inpsList;
    std::vector<Value> outpsList;

    void getSibMods();
    bool processedSiblings = false;

    HWModuleOp analyzedOp;
    OutputOp analyzedOutputOp;
    // mlir::AnalysisManager am;

    const HWPPAData * ppaData;

    bool finished = false;

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

HWModulePPAModel & HWModulePPAModel::getModel(Operation * moduleOp, const HWPPAData * ppaData)
{
    static std::unordered_map<Operation *, HWModulePPAModel> moduleModels;
    static std::unordered_set<Operation *> exists;

    std::mutex mtx;
    std::unique_lock<std::mutex> lock(mtx);

    auto it = exists.find(moduleOp);
    if (it == exists.end())
    {
        exists.insert(moduleOp);
        lock.unlock();
        moduleModels.insert({moduleOp, HWModulePPAModel(moduleOp, ppaData)});
        return moduleModels.at(moduleOp);
    }

    while (moduleModels.find(moduleOp) == moduleModels.end() || !moduleModels.at(moduleOp).finished)
    {
        lock.unlock();
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        lock.lock();
    }
    lock.unlock();

    return moduleModels.at(moduleOp);
}

// HWModulePPAModel & HWModulePPAModel::getModel(Operation * moduleOp)
// {
//     // Maps and synchronization primitives are static so they are shared across
//     // threads. A per-module "constructing" flag + condition_variable is used so
//     // multiple threads can construct different modules in parallel while callers
//     // asking for a module already under construction will wait for completion.
//     static std::mutex gModelsMutex;
//     static std::unordered_map<Operation *, HWModulePPAModel> moduleModels;
//     static std::unordered_set<Operation *> constructing;
//     static std::unordered_map<Operation *, std::unique_ptr<std::condition_variable>> condVars;
//     static std::unordered_map<Operation *, std::thread::id> constructingThread;

//     std::unique_lock<std::mutex> lk(gModelsMutex);

//     // If already constructed, return immediately.
//     auto it = moduleModels.find(moduleOp);
//     if (it != moduleModels.end())
//         return it->second;

//     // If another thread is constructing this module, wait for it to finish.
//     if (constructing.count(moduleOp)) {
//         auto condIt = condVars.find(moduleOp);
//         if (condIt == condVars.end())
//             condIt = (condVars.emplace(moduleOp, std::make_unique<std::condition_variable>())).first;
//         // Wait until the module has been inserted into moduleModels.
//         condIt->second->wait(lk, [&] { return moduleModels.find(moduleOp) != moduleModels.end(); });
//         return moduleModels.find(moduleOp)->second;
//     }

//     // Mark this module as being constructed by this thread.
//     constructing.insert(moduleOp);
//     auto condIt = condVars.find(moduleOp);
//     if (condIt == condVars.end())
//         condIt = (condVars.emplace(moduleOp, std::make_unique<std::condition_variable>())).first;
//     constructingThread[moduleOp] = std::this_thread::get_id();

//     // Unlock while constructing to allow other threads to proceed in parallel.
//     lk.unlock();
//     HWModulePPAModel model(moduleOp);
//     lk.lock();

//     // Insert the constructed model into the shared map.
//     auto inserted = moduleModels.emplace(moduleOp, std::move(model)).first;

//     // Clear constructing flag and notify any waiters.
//     constructing.erase(moduleOp);
//     constructingThread.erase(moduleOp);
//     condIt->second->notify_all();

//     return inserted->second;
// }

struct HWPowerAnalyzePass
    : public circt::hw::impl::HWPowerAnalyzeBase<HWPowerAnalyzePass> {
    using Base::Base;

    void runOnOperation() override;
    void parseVCD();
    void walkOps();
    void calcSwitchingActivity();
    void parsePPAJson(const std::string &jsonPath, HWPPAData &ppaData);

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
    size_t indexFromArgs(uint16_t in1Width, uint16_t auxWidth = 0);

};
}

void HWModulePPAModel::getSibMods()
{
    // llvm::errs() << "[PPA] Getting sibling HWModuleOps " << analyzedOp << '\n';
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

bool HWModulePPAModel::isZeroCostOp(Operation * op)
{
    return llvm::dyn_cast<comb::ConcatOp>(op) ||
           llvm::dyn_cast<comb::ReplicateOp>(op) ||
           llvm::dyn_cast<comb::ReverseOp>(op) ||
           llvm::dyn_cast<comb::ExtractOp>(op) ||
           llvm::dyn_cast<WireOp>(op) ||
           llvm::dyn_cast<OutputOp>(op) ||
           llvm::dyn_cast<HWModuleOp>(op) ||
           llvm::dyn_cast<InstanceOp>(op) ||
           llvm::dyn_cast<BitcastOp>(op) ||
           llvm::dyn_cast<ConstantOp>(op) ||
           llvm::dyn_cast<seq::ConstClockOp>(op) ||
           llvm::dyn_cast<seq::FromClockOp>(op) ||
           llvm::dyn_cast<seq::ToClockOp>(op) ||
           llvm::dyn_cast<seq::InitialOp>(op) ||
           llvm::dyn_cast<seq::FromImmutableOp>(op) ||
           llvm::dyn_cast<seq::YieldOp>(op);
}

HWModulePPAModel::OperandInfo HWModulePPAModel::getOperandInfo(Value operand)
{
    Operation * ownOp = operand.getDefiningOp();

    // Presumably block input
    if (!ownOp)
    {
        return { false, static_cast<int>(operand.getType().getIntOrFloatBitWidth()) };
    }

    if (llvm::dyn_cast<WireOp>(ownOp) || llvm::dyn_cast<comb::ReverseOp>(ownOp))
    {
        return getOperandInfo(ownOp->getOperand(0));
    }
    if (llvm::dyn_cast<comb::ReplicateOp>(ownOp) || llvm::dyn_cast<comb::ExtractOp>(ownOp))
    {
        OperandInfo retVal = getOperandInfo(ownOp->getOperand(0));
        retVal.width = operand.getType().getIntOrFloatBitWidth();
        return retVal;
    }

    OperandInfo retVal;
    retVal.isConstant = false;
    if (!operand.getType().isIntOrIndexOrFloat())
    { 
        retVal.width = 1;
    } else {
        retVal.width = operand.getType().getIntOrFloatBitWidth();
    }

    if (llvm::dyn_cast<ConstantOp>(ownOp))
    {
        retVal.isConstant = true;
        return retVal;
    }

    if (llvm::dyn_cast<comb::ConcatOp>(ownOp))
    {
        int nonConstWidth = retVal.width;
        for (uint i = 0; i < ownOp->getNumOperands(); i++) {
            Value subOperand = ownOp->getOperand(i);
            OperandInfo subOpInfo = getOperandInfo(subOperand);
            if (!subOpInfo.isConstant) { break; }
            nonConstWidth -= subOpInfo.width;
        }
        retVal.width = nonConstWidth;
    }
    return retVal;
}

static HWPPASetPtr zeroedSet = std::make_shared<HWPPASet>(HWPPASet{{
    0,
    0,
    0,
    0,
    0,
    0,
}});

HWPPASetPtr HWModulePPAModel::getPPAInfo(Operation * op)
{
    if (!op) {
        return zeroedSet;
    }
    if (isZeroCostOp(op)) {
        return zeroedSet;
    }

    if (llvm::dyn_cast<comb::AddOp>(op)) 
    { 
        // TECHNICALLY ADDS ARE VARIADIC BUT THEYRE A SINGLE WIDTH IN HWIR SO THIS IS FINE
        // THIS IS WHAT WOULD BE CORRECT FOR FIRRTL AND SINCE THIS MAY BE PORTED TO FIRRTL THIS IS PREFERRED
        // SAME FOR OTHERS VARIADIC OPS HERE
        return ppaData->data.at("add").getBestMatch(
            getOperandInfo(op->getOperand(0)).width,
            getOperandInfo(op->getOperand(1)).width); 
    } 
    if (llvm::dyn_cast<comb::AndOp>(op)) 
    { 
        return ppaData->data.at("and").getBestMatch(1, 0); 
    }
    if (llvm::dyn_cast<comb::DivSOp>(op) || llvm::dyn_cast<comb::DivUOp>(op)) 
    { 
        return ppaData->data.at("div").getBestMatch(
            getOperandInfo(op->getOperand(0)).width,
            getOperandInfo(op->getOperand(1)).width); 
    }
    if (llvm::dyn_cast<comb::ICmpOp>(op) || llvm::dyn_cast<EnumCmpOp>(op)) 
    { 
        return ppaData->data.at("equal").getBestMatch(
            getOperandInfo(op->getOperand(0)).width,
            0); 
    }
    if (llvm::dyn_cast<comb::ModSOp>(op) || llvm::dyn_cast<comb::ModUOp>(op))
    { 
        return ppaData->data.at("mod").getBestMatch(
            getOperandInfo(op->getOperand(0)).width,
            getOperandInfo(op->getOperand(1)).width); 
    }
    if (llvm::dyn_cast<comb::MulOp>(op)) 
    { 
        return ppaData->data.at("mul").getBestMatch(
            getOperandInfo(op->getOperand(0)).width,
            getOperandInfo(op->getOperand(1)).width); 
    }
    if (llvm::dyn_cast<comb::MuxOp>(op)) 
    { 
        return ppaData->data.at("mux").getBestMatch(
            getOperandInfo(op->getOperand(1)).width,
            getOperandInfo(op->getOperand(0)).width); // cond is the 0th operand
    }
    if (llvm::dyn_cast<comb::OrOp>(op)) 
    { 
        return ppaData->data.at("or").getBestMatch(1, 0); 
    }
    if (llvm::dyn_cast<comb::ParityOp>(op) || llvm::dyn_cast<comb::XorOp>(op)) 
    { 
        return ppaData->data.at("xor").getBestMatch(1, 0); 
    }
    // TODO : Strength Reduction
    if (llvm::dyn_cast<comb::ShlOp>(op))
    {
        return ppaData->data.at("shift_l").getBestMatch(
            getOperandInfo(op->getOperand(0)).width,
            getOperandInfo(op->getOperand(1)).width);
    }
    if (llvm::dyn_cast<comb::ShrSOp>(op))
    {
        return ppaData->data.at("shift_r").getBestMatch(
            getOperandInfo(op->getOperand(0)).width,
            getOperandInfo(op->getOperand(1)).width);
    }
    if (llvm::dyn_cast<comb::ShrUOp>(op))
    {
        return ppaData->data.at("shift_br").getBestMatch(
            getOperandInfo(op->getOperand(0)).width,
            getOperandInfo(op->getOperand(1)).width);
    }
    if (llvm::dyn_cast<comb::SubOp>(op))
    { 
        return ppaData->data.at("sub").getBestMatch(
            getOperandInfo(op->getOperand(0)).width,
            getOperandInfo(op->getOperand(1)).width); 
    }
    if (llvm::dyn_cast<comb::TruthTableOp>(op)) 
    { 
        return ppaData->data.at("mux").getBestMatch(
            getOperandInfo(op->getOperand(0)).width,
            1); 
    }

    if (llvm::dyn_cast<seq::ClockGateOp>(op))
    {
        return ppaData->data.at("mux").getBestMatch(
            1,
            1);
    }
    if (llvm::dyn_cast<seq::ClockInverterOp>(op))
    {
        return ppaData->data.at("not").getBestMatch(
            1,
            0);
    }
    if (llvm::dyn_cast<seq::ClockMuxOp>(op))
    {
        return ppaData->data.at("mux").getBestMatch(
            1,
            1);
    }
    if (llvm::dyn_cast<seq::CompRegOp>(op) || llvm::dyn_cast<seq::CompRegClockEnabledOp>(op) ||
        llvm::dyn_cast<seq::FirRegOp>(op)  || llvm::dyn_cast<seq::ShiftRegOp>(op))
    {
        return ppaData->data.at("reg_pos").getBestMatch(
            1,
            0);
    }


    // TODO: HANDLE FIFOs and MEMs
    // seq::FIFOOp
    // seq::FirMemOp
    // seq::FirMemReadOp
    // seq::FirMemReadWriteOp
    // seq::FirMemWriteOp
    // seq::HLMemOp
    // seq::ReadPortOp
    // seq::WritePortOp

    // TODO: HANDLE HW OPS
    // HWGeneratorSchemaOp
    // HWModuleExternOp
    // HWModuleGeneratedOp
    // HierPathOp
    // InstanceChoiceOp
    // TriggeredOp
    // EnumConstantOp
    // AggregateConstantOp
    // StructCreateOp
    // StructExplodeOp
    // StructExtractOp
    // StructInjectOp
    // UnionCreateOp
    // UnionExtractOp
    // TypeScopeOp
    // TypedeclOp
    // ArrayCreateOp
    // ArrayConcatOp
    // ArrayGetOp
    // ArrayInjectOp
    // ArraySliceOp
    // ParamValueOp

    return zeroedSet;
}

// std::optional<Delay> HWModulePPAModel::getMaxDelayRoot2Leaf(Value & root, Value & leaf)
// {

//     DelaysToLeaf & rootBlk = rootBlocks.at(hash_value(root));
//     // HWPathNodePtr & leafBlk = leafBlocks.at(hash_value(leaf))->node;

//     bool found = false;
//     Delay maxDelay{0,0};
//     for (auto & doIt : rootBlk) {
//         Value destNode = std::get<0>(doIt);
//         Delay d = std::get<1>(doIt);
//         if (destNode == leaf) {
//             found = true;
//             if (d.slow > maxDelay.slow)
//                 maxDelay.slow = d.slow;
//             if (d.fast > maxDelay.fast)
//                 maxDelay.fast = d.fast;
//         }
//     }

//     if (found)
//         return maxDelay;
//     return std::nullopt;
// }

// std::optional<Delay> HWModulePPAModel::getMaxDelayLeafFromRoot(Value & root, Value & leaf)
// {
//     DelaysFromRoot & leafBlk = leafBlocks.at(hash_value(leaf));
//     // HWPathNodePtr & rootBlk = rootBlocks.at(hash_value(root))->node;

//     bool found = false;
//     Delay maxDelay{0,0};
//     for (auto & diIt : leafBlk) {
//         Value srcNode = diIt.node. std::get<0>(diIt);
//         Delay d = std::get<1>(diIt);
//         if (srcNode == root) {
//             found = true;
//             if (d.slow > maxDelay.slow)
//                 maxDelay.slow = d.slow;
//             if (d.fast > maxDelay.fast)
//                 maxDelay.fast = d.fast;
//         }
//     }

//     if (found)
//         return maxDelay;
//     return std::nullopt;
// }

HWModulePPAModel & HWModulePPAModel::getModulesAnalysis(InstanceOp & op)
{
    // Get analysis for every sibling HWModuleOp in the circuit:

    if (!processedSiblings) { getSibMods(); }

    auto model = instances.find(op.getModuleName());
    if (model == instances.end())
    {
        model = instances.emplace(
            op.getModuleName(),
            HWModulePPAModel::getModel(siblings.find(op.getModuleName())->second, ppaData)
        ).first;

        // HWModuleOp mop = model->second.analyzedOp;
        // for (Value real : mop->getResults()) {
        
        // }

    }
    return model->second;
}

HWNodeDelaysPtr HWModulePPAModel::dfsPathBackward(Operation * next, int nextInd, HWNodeDelaysPtr & child, int chiInd)
{
    // We traverse forward first, so all nodes should already be in blocks
    Operation * curOp = next;
    if (!curOp)
    {
        curOp = analyzedOp.getOperation();
    }
    int curInd = nextInd;

    auto blk = blocks.find(curOp);

    if (blk == blocks.end())
    {
        HWPathNodePtr pn = std::make_shared<HWPathNode>();
        pn->op = curOp;
        // pn->isTrueRoot = false;
        // pn->isTrueLeaf = false;
        pn->ppaInfo = getPPAInfo(curOp);

        HWNodeDelaysPtr newBlock = std::make_shared<HWNodeDelays>();
        newBlock->node = pn;
        newBlock->forwardIndices = llvm::SmallVector<int>{};
        newBlock->backwardIndices = llvm::SmallVector<int>{};

        blk = blocks.insert({curOp, newBlock}).first;
    }
    HWNodeDelaysPtr & cur = blk->second;


    if ((child != nullptr) && (isa<HWModuleOp>(curOp) || isa<seq::Clocked>(curOp) || isa<ConstantOp>(curOp))) {
        HWPathNodePtr pn = cur->node;
        // HWPPAInfo outpPPA = getPPAInfo(curOp);
        bool isTrue = (isa<seq::Clocked>(curOp) || isa<ConstantOp>(curOp));
        if (!cur->delaysIn.size())
        {
            
            cur->delaysIn.push_back(DelayPath{pn, 0, isTrue, -1, -1});
            
        }
        // child->delaysIn.push_back(DelayPath{pn, Delay{pn->ppaInfo.slowDelay, pn->ppaInfo.fastDelay}, curInd, chiInd});
        // pn->isTrueRoot |= (isa<seq::Clocked>(curOp) || isa<ConstantOp>(curOp));
        return cur;
    }

    if (cur->backwardIndices.size() < curOp->getNumOperands()) 
    {
    
        for (int i = 0; i < curOp->getNumOperands(); i++)
        {
            if (std::find(cur->backwardIndices.begin(), cur->backwardIndices.end(), i) != cur->backwardIndices.end())
            {
                continue;
            }
            cur->backwardIndices.push_back(i);

            Value operand = curOp->getOperand(i);
            Operation * defOp = operand.getDefiningOp();
            OpResult res = dyn_cast<OpResult>(operand);

            HWNodeDelaysPtr par = dfsPathBackward(defOp, i, cur, curInd);

            if (defOp && isa<InstanceOp>(defOp))
            {
                // llvm::errs() << "WE GOT HERE TEAM! curop is " << *curOp << " operand: " << i << " from result: " << res.getResultNumber() << "\n";

                InstanceOp instOp = dyn_cast<InstanceOp>(defOp);
                HWModulePPAModel & model = getModulesAnalysis(instOp);
                HWNodeDelaysPtr & outp = model.blocks.at(model.analyzedOutputOp.getOperation());

                for (auto & pathInternal : outp->delaysIn)
                {
                    // llvm::errs() << pathInternal.node->op->getName() << " " << pathInternal.isTrue << " " << pathInternal.toInd << " " << pathInternal.fromInd << " " << '\n';
                    if (pathInternal.toInd != res.getResultNumber())
                    {
                        continue;
                    }
                    if (pathInternal.isTrue)
                    {
                        double pathInternalD = pathInternal.delay;

                        cur->delaysIn.push_back(DelayPath{par->node, pathInternalD, true, pathInternal.toInd, i});
                        continue;
                    }
                    for (auto & po : par->delaysIn)
                    {
                        if ((pathInternal.fromInd != po.toInd))
                        {
                            continue;
                        }
                        double pathInternalD = pathInternal.delay;
                        double poD = po.delay;

                        cur->delaysIn.push_back(DelayPath{po.node, pathInternalD + poD, po.isTrue, po.fromInd, i});
                    }

                }

                continue;
            }
            if (defOp && (isa<HWModuleOp>(defOp) || isa<seq::Clocked>(defOp) || isa<ConstantOp>(defOp)))
            {
                double curDelay = par->node->ppaInfo->begin()->delay;
                bool isTrue = (isa<seq::Clocked>(defOp) || isa<ConstantOp>(defOp));
                cur->delaysIn.push_back(DelayPath{par->node, curDelay, isTrue, static_cast<int>(res.getResultNumber()), i});
                
                continue;
            }
            for (auto & pi : par->delaysIn)
            {
                double curDelay = par->node->ppaInfo->begin()->delay;
                double piD = pi.delay;
                cur->delaysIn.push_back(DelayPath{pi.node, piD + curDelay, pi.isTrue, pi.fromInd, i});
            }
        }
    }
    return cur;

}

// HWNodeDelaysPtr HWModulePPAModel::dfsPathBackward(Value & next, int nextInd, HWNodeDelaysPtr & child, int chiInd)
// {
//     // We traverse forward first, so all nodes should already be in blocks
//     Value & curVal = next;
//     int curInd = nextInd;
//     Operation * defOp = curVal.getDefiningOp();
//     if (isa<BlockArgument>(curVal)) {
//         defOp = analyzedOp.getOperation();
//     }
//     auto blk = blocks.find(hash_value(curVal));
    
//     if (blk == blocks.end())
//     {
//         HWPathNodePtr pn = std::make_shared<HWPathNode>();
//         pn->op = defOp;
//         pn->val = curVal;
//         pn->isTrueRoot = false;
//         pn->isTrueLeaf = false;
//         pn->ppaInfo = getPPAInfo(defOp);

//         HWNodeDelaysPtr newBlock = std::make_shared<HWNodeDelays>();
//         newBlock->node = pn;

//         blk = blocks.insert({hash_value(curVal), newBlock}).first;
//     }
//     HWNodeDelaysPtr & cur = blk->second;

//     if (allOps.find(defOp) == allOps.end())
//     {
//         allOps.insert({defOp, llvm::SmallVector<HWNodeDelaysPtr>()});
//     }
//     allOps.at(defOp).push_back(cur);

//     // If we haven't traversed this node before, traverse it
//     if (!cur->delaysIn.size()) {

//         // Check if this is a root node that we are dealing with
//         if (dyn_cast<HWModuleOp>(defOp) || dyn_cast<seq::Clocked>(defOp) || dyn_cast<ConstantOp>(defOp)) {
//             HWPathNodePtr pn = blk->second->node;
//             cur->delaysIn.emplace(pn, Delay{pn->ppaInfo.slowDelay,pn->ppaInfo.fastDelay});
//             pn->isTrueRoot = (dyn_cast<seq::Clocked>(defOp) || dyn_cast<ConstantOp>(defOp));
//         }
//         else if (InstanceOp instOp = dyn_cast<InstanceOp>(defOp))
//         {
//             HWModulePPAModel & model = getModulesAnalysis(instOp);
//             Value & output = model.outpsList.at(curInd);
//             HWNodeDelaysPtr paths = model.leafBlocks.find(hash_value(output))->second;
//             for (auto & pi : paths->delaysIn) {
//                 if (!pi.first->isTrueRoot)
//                 {
//                     continue;
//                 }
//                 Delay piD = pi.second;
//                 HWPathNodePtr pn = cur->node;
//                 pn->isTrueRoot = true;
//                 cur->delaysIn.emplace(pn, Delay{piD.slow, piD.fast});
//             }

//             for (uint i = 0; i < defOp->getNumOperands(); i++)
//             {

//                 auto cp = model.getMaxDelayLeafFromRoot(model.inpsList.at(i), output);
//                 if (!cp.has_value()) {
//                     // No path between root and leaf in the instance
//                     continue;
//                 }
//                 Value operand = defOp->getOperand(i);
//                 HWNodeDelaysPtr par = dfsPathBackward(operand, i,  cur, curInd);
//                 for (auto & pi : par->delaysIn)
//                 {
//                     int curSlow = cp->slow;
//                     int curFast = cp->fast;
//                     Delay piD = pi.second;
//                     cur->delaysIn.emplace(pi.first, Delay{piD.slow + curSlow, piD.fast + curFast });
//                 }
//             }
//         }
//         else // Otherwise, continue traversing
//         {
//             for (uint i = 0; i < defOp->getNumOperands(); i++)
//             {
//                 Value operand = defOp->getOperand(i);
//                 HWNodeDelaysPtr par = dfsPathBackward(operand, i, cur, curInd);
//                 for (auto & pi : par->delaysIn)
//                 {
//                     int curSlow = cur->node->ppaInfo.slowDelay;
//                     int curFast = cur->node->ppaInfo.fastDelay;
//                     Delay piD = pi.second;
//                     cur->delaysIn.emplace(pi.first, Delay{piD.slow + curSlow, piD.fast + curFast });
//                 }
//             }
//         }
//     }

//     // if (child == nullptr)
//     //     return cur;

//     // // Give delayIn info to the current node's parent
//     // int curSlow = cur->node->ppaInfo.slowDelay;
//     // int curFast = cur->node->ppaInfo.fastDelay;
//     // for (auto & pi : cur->delaysIn)
//     // {
//     //     Delay piD = pi.second;
//     //     child->delaysIn.emplace(pi.first, Delay{piD.slow + curSlow, piD.fast + curFast });
//     // }
//     return cur;
// }


// HWNodeDelaysPtr HWModulePPAModel::dfsPathForward(Value & next, int nextInd, HWNodeDelaysPtr & parent, int parInd)
// {
//     // Rename
//     Value & curVal = next;
//     int curInd = nextInd;
//     Operation * defOp = curVal.getDefiningOp();
//     if (isa<BlockArgument>(curVal)) {
//         defOp = analyzedOp.getOperation();
//     }

//     auto blk = blocks.find(hash_value(curVal));

//     // If we haven't traversed this node before, construct a block add it to blocks
//     if (blk == blocks.end())
//     {
//         HWPathNodePtr pn = std::make_shared<HWPathNode>();
//         pn->op = defOp;
//         pn->val = curVal;
//         pn->isTrueRoot = false;
//         pn->isTrueLeaf = false;
//         pn->ppaInfo = getPPAInfo(defOp);

//         HWNodeDelaysPtr newBlock = std::make_shared<HWNodeDelays>();
//         newBlock->node = pn;

//         blk = blocks.insert({hash_value(curVal), newBlock}).first;
//     }
//     HWNodeDelaysPtr & cur = blk->second;

//     if (allOps.find(defOp) == allOps.end())
//     {
//         allOps.insert({defOp, llvm::SmallVector<HWNodeDelaysPtr>()});
//     }
//     allOps.at(defOp).push_back(cur);

//     if (!cur->delaysOut.size())
//     {
//         // Now traverse the new block (the current block)
//         for (auto & use : curVal.getUses()) {

//             Operation * owner = use.getOwner();
            
//             if (InstanceOp instOp = dyn_cast<InstanceOp>(owner))
//             {
//                 HWModulePPAModel & model = getModulesAnalysis(instOp);
                
//                 Value & input = model.inpsList.at(use.getOperandNumber());
//                 HWNodeDelaysPtr paths = model.rootBlocks.find(hash_value(input))->second;
//                 for (auto & po : paths->delaysOut) {
//                     if (!po.first->isTrueLeaf)
//                     {
//                         continue;
//                     }
//                     Delay poD = po.second;
//                     HWPathNodePtr pn = cur->node;
//                     pn->isTrueLeaf = true;
//                     cur->delaysOut.emplace(pn, Delay{poD.slow, poD.fast});
//                 }

//                 for (uint i = 0; i < owner->getNumResults(); i++)
//                 {

//                     auto cp = model.getMaxDelayRoot2Leaf(input, model.outpsList.at(i));
//                     if (!cp.has_value()) {
//                         // No path between root and leaf in the instance
//                         continue;
//                     }
//                     OpResult res = owner->getResult(i);
//                     dfsPathForward(res, i, cur, curInd);
//                 }
//             }

//             // Check if this use of cur is a leaf node (output or clocked op)
//             else if (isa<OutputOp>(owner) || isa<seq::Clocked>(owner)) {
//                 HWPathNodePtr pn = blk->second->node;
//                 HWPPAInfo outpPPA = getPPAInfo(owner);
//                 cur->delaysOut.emplace(pn, Delay{outpPPA.slowDelay,outpPPA.fastDelay});
//                 pn->isTrueLeaf = dyn_cast<seq::Clocked>(owner);
//                 foundLeaves++;

//                 if (allOps.find(owner) == allOps.end())
//                 {
//                     allOps.insert({owner, llvm::SmallVector<HWNodeDelaysPtr>()});
//                 }

//             } else
//             { // Otherwise, continue traversing
//                 for (uint i = 0; i < owner->getNumResults(); i++)
//                 {

//                     OpResult res = owner->getResult(i);
//                     dfsPathForward(res, i, cur, curInd);
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

HWNodeDelaysPtr HWModulePPAModel::dfsPathForward(Operation * next, int nextInd, HWNodeDelaysPtr & parent, int parInd)
{
    // Rename
    Operation * curOp = next;
    int curInd = nextInd;

    auto blk = blocks.find(curOp);

    // If we haven't traversed this node before, construct a block add it to blocks
    if (blk == blocks.end())
    {
        HWPathNodePtr pn = std::make_shared<HWPathNode>();
        pn->op = curOp;
        // pn->isTrueRoot = false;
        // pn->isTrueLeaf = false;
        pn->ppaInfo = getPPAInfo(curOp);

        HWNodeDelaysPtr newBlock = std::make_shared<HWNodeDelays>();
        newBlock->node = pn;
        newBlock->forwardIndices = llvm::SmallVector<int>{};
        newBlock->backwardIndices = llvm::SmallVector<int>{};

        blk = blocks.insert({curOp, newBlock}).first;
    }
    HWNodeDelaysPtr & cur = blk->second;

    if ((parent != nullptr) && (isa<OutputOp>(curOp) || isa<seq::Clocked>(curOp))) {
        HWPathNodePtr pn = cur->node;
        // HWPPAInfo outpPPA = getPPAInfo(curOp);
        bool isTrue = isa<seq::Clocked>(curOp);
        if (!cur->delaysOut.size())
        {
            cur->delaysOut.push_back(DelayPath{pn, 0, isTrue, -1, -1});
        }
        parent->delaysOut.push_back(DelayPath{pn, pn->ppaInfo->begin()->delay, isTrue, parInd, curInd});
        // pn->isTrueLeaf |= isa<seq::Clocked>(curOp);
        foundLeaves++;
        return cur;
    }

    // If there's still results we haven't traversed, continue traversing
    // This will happen when an op is visited the first time and possibly 
    // if it's revisisted WHILE it traverses other paths
    HWModuleOp mop = llvm::dyn_cast<HWModuleOp>(curOp);
    int numResults = mop ? mop.getNumInputPorts() : curOp->getNumResults();
    if (cur->forwardIndices.size() < numResults)
    {
        for (int i = 0; i < numResults; i++)
        {
            // If we've already traversed this result, don't do so again
            // This prevents infinite loops
            // It's okay if we get back here though, it just means that theres a path
            // which we are traversing which passes through the module (this doesn't work
            // for combinational loops but we assume that those don't exist in hw modules)
            if (std::find(cur->forwardIndices.begin(), cur->forwardIndices.end(), i) != cur->forwardIndices.end())
            {
                continue;
            }
            cur->forwardIndices.push_back(i);

            auto resultUses = mop ? mop.getArgumentForInput(i).getUses() : curOp->getResult(i).getUses();
            for (auto & use : resultUses)
            {
                Operation * owner = use.getOwner();
                dfsPathForward(owner, use.getOperandNumber(), cur, i);
            }
        }
    }
    if (parent == nullptr)
        return cur;


    if (InstanceOp instOp = dyn_cast<InstanceOp>(curOp))
    {
        HWModulePPAModel & model = getModulesAnalysis(instOp);
        HWNodeDelaysPtr & root = model.blocks.at(model.analyzedOp.getOperation());
        // Value & input = model.inpsList.at(curInd);

        // Paths from root of instance to leaves of instance
        for (auto & pathInternal : root->delaysOut)
        {
            if (pathInternal.fromInd != curInd)
            {
                continue;
            }
            if (pathInternal.isTrue)
            {
                double pathInternalD = pathInternal.delay;

                // This is a path into the module
                parent->delaysOut.push_back(DelayPath{cur->node, pathInternalD, true, parInd, pathInternal.fromInd});
                continue;
            }
            
            // Paths from leaves of instance to leaves of the module
            for (auto & po : cur->delaysOut)
            {
                if ((pathInternal.toInd != po.fromInd))
                {
                    continue;
                }
                double pathInternalD = pathInternal.delay;
                double poD = po.delay;

                // This is a path through the module
                parent->delaysOut.push_back(DelayPath{po.node, pathInternalD + poD, po.isTrue, parInd, po.toInd});
            }
        }
        return cur;
    }

    double curDelay = cur->node->ppaInfo->begin()->delay; 
    for (auto & po : cur->delaysOut) {
        double poD = po.delay;
        int ind = po.toInd;
        parent->delaysOut.push_back(DelayPath{po.node, poD + curDelay, po.isTrue, parInd, ind});
    }

    return cur;
}

// void HWModulePPAModel::traverseFromLeaf(Value & leaf, int ind) {

//     if (leafBlocks.find(hash_value(leaf)) != leafBlocks.end())
//     {
//         return;
//     }

//     HWNodeDelaysPtr nullPtr = nullptr;
//     HWNodeDelaysPtr leafBlock = dfsPathBackward(leaf, 0, nullPtr, -1);
//     leafBlocks.emplace(hash_value(leaf), leafBlock);
// };

void HWModulePPAModel::traverseFromRoot(Value & root, int ind) {

    auto rootBlk = blocks.find(analyzedOp.getOperation());
    if (rootBlk == blocks.end())
    {
        HWPathNodePtr pn = std::make_shared<HWPathNode>();
        pn->op = analyzedOp.getOperation();
        // pn->isTrueRoot = false;
        // pn->isTrueLeaf = false;
        pn->ppaInfo = getPPAInfo(analyzedOp.getOperation());

        HWNodeDelaysPtr newBlock = std::make_shared<HWNodeDelays>();
        newBlock->node = pn;
        newBlock->forwardIndices = llvm::SmallVector<int>{};
        newBlock->backwardIndices = llvm::SmallVector<int>{};

        rootBlk = blocks.insert({analyzedOp.getOperation(), newBlock}).first;
    }
    HWNodeDelaysPtr & cur = rootBlk->second;

    if (std::find(cur->forwardIndices.begin(), cur->forwardIndices.end(), ind) != cur->forwardIndices.end()) 
    {
        return;
    }
    cur->forwardIndices.push_back(ind);
    for (auto & use : root.getUses())
    {
        Operation * owner = use.getOwner();
        dfsPathForward(owner, use.getOperandNumber(), cur, ind);
    }
};

std::mutex debugWrLock;

HWModulePPAModel::HWModulePPAModel(Operation * moduleOp, const HWPPAData * ppaDataPtr) : ppaData(ppaDataPtr)
{

    HWModuleOp mop = dyn_cast<HWModuleOp>(moduleOp);
    if (!mop) {
        return;
    }
    // llvm::errs() << "MOP ADDRESS" << mop.getOperation() << '\n';
    // llvm::errs() << "PreMOP Address" << moduleOp << '\n';
    analyzedOp = mop;

    for (size_t i = 0; i < mop.getNumInputPorts(); i++)
    {
        BlockArgument res = mop.getArgumentForInput(i);
        inpsList.push_back(static_cast<Value>(res));
    }

    HWNodeDelaysPtr nullPtr = nullptr;
    dfsPathForward(analyzedOp, -1, nullPtr, -1);

    mop->walk([&](Operation * op) {

        seq::Clocked clockedOp = dyn_cast<seq::Clocked>(op);
        if (clockedOp)
        {
            dfsPathForward(op, -1, nullPtr, -1);
            dfsPathBackward(op, -1, nullPtr, -1);
            return;
        }
        ConstantOp constOp = dyn_cast<ConstantOp>(op);
        if (constOp)
        {
            dfsPathForward(op, -1, nullPtr, -1);
            return;
        }
        OutputOp outputOp = dyn_cast<OutputOp>(op);
        if (outputOp)
        {
            this->analyzedOutputOp = outputOp;
            dfsPathBackward(op, -1, nullPtr, -1);
            return; 
        }
    });

    
    // llvm::errs() << "[PPA] have " << blocks.size() << " blocks\n";
    // llvm::errs() << "[PPA] have " << foundLeaves << " found leaves\n";
    // llvm::errs() << "[PPA] have " << leafBlocks.size() << " blocks leaves\n";
    // for (auto & lb : leafBlocks)
    // {
    //     llvm::errs() << lb.second->node->val << '\n';
    // }
    // llvm::errs() << "[PPA] have " << rootBlocks.size() << " blocks roots\n";
    // for (auto & rb : rootBlocks)
    // {
    //     llvm::errs() << rb.second->node->val << '\n';
    // }

    // for (auto & rb : rootBlocks)
    // {
    //     for (auto & lb : leafBlocks)
    //     {
    //         std::optional<Delay> d = getMaxDelayRoot2Leaf(rb.second->node->val, lb.second->node->val);
    //         if (d)
    //         {
    //             llvm::errs() << "[PPA] Max delay from root " << rb.second->node->val << " to leaf " << lb.second->node->val << " is slow: " << d->slow << ", fast: " << d->fast << '\n';
    //         }
    //         d = getMaxDelayLeafFromRoot(rb.second->node->val, lb.second->node->val);
    //         if (d)
    //         {
    //             llvm::errs() << "[PPA] Max delay to leaf " << lb.second->node->val << " from root " << rb.second->node->val << " is slow: " << d->slow << ", fast: " << d->fast << '\n';
    //             llvm::errs() << "\n";
    //         }
    //     }
    // }

    debugWrLock.lock();

    llvm::errs() << "[PPA] Module " << mop.getName() << '\n';

    for (auto & block : blocks) {

        if (isa<HWModuleOp>(block.second->node->op)) {
            llvm::errs() << "[PPA] Node Block input\n";
        } else {
            llvm::errs() << "[PPA] Node " << *block.second->node->op << '\n';
        }
        // if (block.second->node->op)
        // {
        //     block.second->node->op->dumpPretty();
        // }
        // else {
        //     llvm::errs() << "mod inp";
        // }

        for (auto & pi : block.second->delaysIn) {
            if (isa<HWModuleOp>(pi.node->op)) {
                llvm::errs() << "\n    trueroot: " << pi.isTrue << "    delayIn from module input is: " << pi.delay << '\n';
                continue;
            }
            llvm::errs() << "\n    trueroot: " << pi.isTrue << "    delayIn from " << *pi.node->op;
            // if (pi.first->op) { 
            //     pi.first->op->dumpPretty();
            // } else {
            //     llvm::errs() << "mod inp";
            // }
            llvm::errs() << " is: " << pi.delay << '\n';
        }
        for (auto & po : block.second->delaysOut) {
            llvm::errs() << "\n    trueleaf: " << po.isTrue << "    delayOut to " << *po.node->op;
            // if (po.first->op) {
            //     po.first->op->dumpPretty();
            // } else {
            //     llvm::errs() << "mod out";
            // }
            llvm::errs() << " is: " << po.delay << '\n';
        }
    }
    llvm::errs() << "\n\n\n";

    debugWrLock.unlock();
    
    // debugWrLock.unlock();


    // (2) run DFS a second time to annotate all nodes with the inputs they depend on
    // (3) Go through all nodes in topological order, assigning a fast and slow hw block to each op, and calculating slow and fast path lengths from roots to leaves.


    // Clocked objects have results which are roots and inputs which are leaves

    finished = true;

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

size_t HWPowerAnalyzePass::indexFromArgs(uint16_t in1Width, uint16_t auxWidth)
{
    return static_cast<size_t>(in1Width) << 16 | static_cast<size_t>(auxWidth);
}

void HWPowerAnalyzePass::parsePPAJson(const std::string &jsonPath, HWPPAData &ppaData) {
    std::ifstream in(jsonPath);
    if (!in) {
        llvm::errs() << "[HWPowerAnalyze] Could not open JSON file: " << jsonPath << "\n";
        return;
    }
    std::string jsonText((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    llvm::Expected<llvm::json::Value> parsed = llvm::json::parse(jsonText);
    if (!parsed) {
        llvm::errs() << "[HWPowerAnalyze] Failed to parse JSON: " << jsonPath << "\n";
        llvm::handleAllErrors(parsed.takeError(), [](const llvm::ErrorInfoBase &EIB) {
            EIB.log(llvm::errs());
        });
        llvm::errs() << "\n";
        return;
    }
    llvm::json::Object *rootObj = parsed->getAsObject();
    if (!rootObj) {
        llvm::errs() << "[HWPowerAnalyze] JSON root is not an object\n";
        return;
    }
    for (auto &topKV : *rootObj) {
        const std::string &topKey = topKV.first.str();
        ppaData.data.emplace(topKey, HWPPAGroup());
        HWPPAGroup & grp = ppaData.data[topKey];
        const llvm::json::Array *arr = topKV.second.getAsArray();
        if (!arr) continue;
        std::map<size_t, HWPPASetPtr> infos;
        // 
        for (const auto &elem : *arr) {
            const llvm::json::Object *obj_par = elem.getAsObject();
            const llvm::json::Object *obj = obj_par->getObject("properties");
            if (!obj) continue;
            HWPPAInfo info;
            auto parseDouble = [](const llvm::json::Object *obj, const char *key) -> double {
                // Prefer numeric if present
                if (auto v = obj->getNumber(key))
                    return *v;
                if (auto i = obj->getInteger(key))
                    return static_cast<double>(*i);
                // Fallback: parse string possibly with unit suffixes
                if (auto s = obj->getString(key)) {
                    std::string tmp = s->str();
                    const char *begin = tmp.c_str();
                    char *end = nullptr;
                    double val = std::strtod(begin, &end);
                    if (end != begin)
                        return val; // parsed at least something
                }
                return 0.0;
            };
            info.delay = parseDouble(obj, "Delay");
            info.area = parseDouble(obj, "Area");
            info.lkgpwr = parseDouble(obj, "LkgPwr");
            double swg = parseDouble(obj, "SwgEng");
            double inte = parseDouble(obj, "IntEng");
            info.dynengy = swg + inte;
            
            // TODO GRAB WIDTH FROM JSON
            size_t width = static_cast<size_t>(parseDouble(obj, "width"));
            size_t auxWidth = 0;

            if (!width) 
            {
                width = static_cast<size_t>(parseDouble(obj, "width_in1"));
                auxWidth = static_cast<size_t>(parseDouble(obj, "width_in2"));
            }

            if (!width)
            {
                width = static_cast<size_t>(parseDouble(obj, "in_width"));
                auxWidth = static_cast<size_t>(parseDouble(obj, "aux_width"));
            }

            info.in1Width = width;
            info.auxWidth = auxWidth;
            grp.addInfo(info);
        }
    }
}

void HWPowerAnalyzePass::runOnOperation() 
{
    llvm::Timer watch;
    watch.startTimer();

    HWPPAData * ppaData = new HWPPAData();
    parsePPAJson("/scratch/jtoubes/chisel-learning/carbon/asap7-dc-catapult.json", *ppaData);

    // Test prints: print first 10 instances for each top-level key
    for (const auto &kv : ppaData->data) {
        llvm::errs() << "[PPA JSON] Key: " << kv.first << "\n";
        int count = 0;
        llvm::errs() << "[PPA JSON] InfoSetVec size: " << kv.second.m_infoSets.size() << "\n";
        for (const auto &infoSetVec : kv.second.m_infoSets) {
            llvm::errs() << "[PPA JSON] InfoSetVec size: " << infoSetVec.size() << "\n";
            for (auto & infoSet : infoSetVec) {
                llvm::errs() << "[PPA JSON] InfoSet size: " << infoSet->size() << "\n";
                for (auto & info : *infoSet) {
                    llvm::errs() << "  Instance " << count << ": delay=" << info.delay
                        << ", area=" << info.area
                        << ", lkgpwr=" << info.lkgpwr
                        << ", dynengy=" << info.dynengy << "\n";
                }
                if (++count >= 10) break;
            }
        }
    }

    llvm::errs() << "[PowerAnalyze] Running power analysis pass\n";
    llvm::errs().flush();



    mlir::ModuleOp topOp = getOperation();

    llvm::SmallVector<std::thread *> tPool;
    topOp->walk([&](HWModuleOp modOp) {

        tPool.push_back(new std::thread(HWModulePPAModel::getModel, modOp, ppaData));

        // HWModulePPAModel & a = HWModulePPAModel::getModel(modOp);

    });

    for (std::thread * t : tPool) {
        t->join();
    }
    watch.stopTimer();
    llvm::errs() << "[PowerAnalyze] TIMING INFO\n"; 
    llvm::errs() << "[PowerAnalyze] system time" << watch.getTotalTime().getSystemTime() << "\n";
    llvm::errs() << "[PowerAnalyze] user time" << watch.getTotalTime().getUserTime() << "\n";
    llvm::errs() << "[PowerAnalyze] process time" << watch.getTotalTime().getProcessTime() << "\n";
    llvm::errs() << "[PowerAnalyze] wall time" << watch.getTotalTime().getWallTime() << "\n";

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



    delete ppaData;

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
//===- PowerAnalyze.cpp - Print FIRRTL module names ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the PowerAnalyze pass. For now, it just prints the names of
// all modules in the FIRRTL circuit.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "circt/Dialect/FIRRTL/Passes.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TableGen/TableGenBackend.h"
#include <fstream>
#include <ostream>

namespace circt {
namespace firrtl {
#define GEN_PASS_DEF_POWERANALYZE
#include "circt/Dialect/FIRRTL/Passes.h.inc"
} // namespace firrtl
} // namespace circt

using namespace circt;
using namespace firrtl;
using namespace llvm::TableGen::Emitter;

namespace {

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

struct PowerAnalyzePass
    : public circt::firrtl::impl::PowerAnalyzeBase<PowerAnalyzePass> {
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
} // end anonymous namespace


void PowerAnalyzePass::parseVCDVar(VCDData & v, std::istringstream & iss)
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
        ptr = std::make_shared<SignalInfo>(SignalInfo{name, width});
        symbolToSignal[symbol] = ptr;
    } else
    {
        ptr = it->second;
    }
    hierarchyToSignal[fullName] = ptr;
}

bool PowerAnalyzePass::parseVCDHeaderLine(VCDData & v, std::istringstream & iss)
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

void PowerAnalyzePass::parseVCDBooleanLine(VCDData & v, std::string & token)
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

void PowerAnalyzePass::parseVCDBitvectorLine(VCDData & v, std::istringstream & iss, std::string & token)
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

void PowerAnalyzePass::parseVCD()
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

void PowerAnalyzePass::calcSwitchingActivity()
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

void PowerAnalyzePass::walkOps()
{

    CircuitOp circuitOp = getOperation();
    llvm::errs() << "[PowerAnalyze] Analyzing circuit: " << circuitOp.getOperationName() << "\n";

    // Print the operation name for each operation in the circuit
    circuitOp->walk([&](Operation *op) {
        llvm::errs() << "[PowerAnalyze] Found operation: " << op->getName() << "\n";
    });

    // circuitOp->walk(FnT &&callback);
}


void PowerAnalyzePass::runOnOperation() 
{

    // Print the VCD file option if provided
    if (!powerAnalyzeVcd.empty()) {
        llvm::errs() << "[PowerAnalyze] Using VCD file for switching activity: "
                     << powerAnalyzeVcd << "\n";
    }
    // circuitOp->dumpPretty();
    llvm::errs() << "[PowerAnalyze] FIRRTL Modules found:\n";

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

    // Let's parse this VCD :sunglasses:
    if (!powerAnalyzeVcd.empty()) {
        parseVCD();
    }
    calcSwitchingActivity();
    walkOps();

    markAllAnalysesPreserved();
}
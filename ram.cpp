#include <iostream>
#include <unordered_map>
#include <utility>
#include "ram.h"

Ram::Ram() {}

void Ram::print() const {
    for (const auto& entry : wentry) {
        std::cout << "Key: " << entry.first << ", Value: (" 
                    << entry.second.first << ", " 
                    << entry.second.second << ")\n";
    }
}

// Get entry by key, returns a pair of (0, 0) if key is not found
std::pair<double, double> Ram::getEntry(int key) {
    auto it = wentry.find(key);
    if (it != wentry.end()) {
        return it->second;
    } else {
        return std::make_pair(0.0, 0.0);
    }
}

// Update entry, incrementing the first element of the pair by 1
// and adding the value to the second element, or inserting a new entry
void Ram::updEntry(int key, double value) {
    auto it = wentry.find(key);
    if (it != wentry.end()) {
        it->second.first += 1.0;
        it->second.second += value;
    } else {
        wentry[key] = std::make_pair(1.0, value);
    }
}

WRam::WRam() {}

void WRam::print() const {
    for (const auto& entry : wentry) {
        std::cout << "Key: " << entry.first << ", Value: " 
                    << entry.second << ")\n";
    }
}

// Get entry by key, returns a pair of (0, 0) if key is not found
double WRam::getEntry(int key) {
    auto it = wentry.find(key);
    if (it != wentry.end()) {
        return it->second;
    } else {
        return 0.0;
    }
}

// Update entry, incrementing the first element of the pair by 1
// and adding the value to the second element, or inserting a new entry
void WRam::updEntry(int key) {
    auto it = wentry.find(key);
    if (it != wentry.end()) {
        it->second += 1.0;
    } else {
        wentry[key] = 1.0;
    }
}

int main() {
    Ram ram;

    // Testing the Ram class
    ram.updEntry(1, 10.0);
    ram.updEntry(2, 20.0);
    ram.updEntry(1, 5);

    ram.print(); // Prints out the contents of the wentry

    // Accessing an entry
    auto entry = ram.getEntry(1);
    std::cout << "Entry for key 1: (" << entry.first << ", " << entry.second << ")\n";

    return 0;
}

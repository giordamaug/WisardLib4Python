#ifndef RAM_H  // Include guard to prevent multiple inclusions
#define RAM_H

#include <unordered_map>
#include <utility>

class Ram {
public:
    // Constructor
    Ram();

    // Method to print the contents of the map
    void print() const;

    // Method to get entry by key, returns a pair (0,0) if key is not found
    std::pair<double, double> getEntry(int key);

    // Method to update entry, increment first element and add value to second element
    void updEntry(int key, double value);

private:
    // Public data member to store the entries
    std::unordered_map<int, std::pair<double, double> > wentry;
};

class WRam {
    public:
        // Constructor
        WRam();
    
        // Method to print the contents of the map
        void print() const;
    
        // Method to get entry by key, returns a pair (0,0) if key is not found
        double getEntry(int key);
    
        // Method to update entry, increment first element and add value to second element
        void updEntry(int key);
    
    private:
        // Public data member to store the entries
        std::unordered_map<int, double> wentry;
    };

#endif // RAM_H

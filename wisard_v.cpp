#include <pybind11/stl.h>
#include <pybind11/complex.h>
#include <pybind11/functional.h>
#include <pybind11/chrono.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <random>
#include <vector>

namespace py = pybind11;

class WiSARDreg {
    private:
        int _nobits, _retina_size, _nrams, _nloc;
        std::vector<int> _mapping;
        std::vector<int> _revmapping;
        std::vector<std::vector<std::pair<float,float>>> _rams;
        std::vector<float> _mi;
        float _maxvalue;
        int _traincount;    
    
    public:
        // Initialize mypowers (powers of 2)
        std::vector<int> mypowers = []() {
            std::vector<int> result(32);
            for (int i = 0; i < 32; ++i) {
                result[i] = static_cast<int>(std::pow(2, i));
            }
            return result;
        }();
    
        WiSARDreg(int size, int n_bits = 16, int map = -1)
            : _nobits(n_bits), _retina_size(size) {
    
            _nloc = 1 << _nobits;
            _nrams = (_retina_size % _nobits == 0) ? (_retina_size / _nobits) : (_retina_size / _nobits + 1);
            _mapping.resize(_retina_size);
            _revmapping.resize(_retina_size);
    
            for (int i = 0; i < _retina_size; ++i) {
                _mapping[i] = i;
            }
    
            if (map > -1) {
                std::mt19937 rng(map);  // Initialize random number generator with seed
                std::shuffle(_mapping.begin(), _mapping.end(), rng);  // Use std::shuffle for reproducibility
            
                for (int i = 0; i < _retina_size; ++i) {
                    _revmapping[_mapping[i]] = i;
                }
            }
    
            _mi = std::vector<float>(_retina_size, 0.0);
            _maxvalue = 0;
            _traincount = 0;
            _rams = std::vector<std::vector<std::pair<float,float>>>(_nrams, std::vector<std::pair<float,float>>(_nloc, {0.0, 0.0}));
        }
                        
        std::vector<int> _mk_tuple_float(py::array_t<float> X, int ntics, py::array_t<float> offsets, py::array_t<float> ranges) {
            auto X_buf = X.request();
            float* ptr = static_cast<float*>(X_buf.ptr);
            auto off_buf = offsets.request();
            float* off_ptr = static_cast<float*>(off_buf.ptr);
            auto rng_buf = ranges.request();
            float* rng_ptr = static_cast<float*>(rng_buf.ptr);
            std::vector<int> intuple(_nrams, 0);
            
            for (int i = 0; i < _nrams; i++) {
                for (int j = 0; j < _nobits; j++) {
                    int x = _mapping[((i * _nobits) + j) % _retina_size];
                    int index = x / ntics;
                    int value = int((ptr[index] - off_ptr[index]) * ntics / rng_ptr[index]);
                    intuple[i] += (x % ntics < value) ? (1 << (_nobits - 1 - j)) : 0;
                }
            }
            return intuple;
        }

        void train_tpl_val(const std::vector<int>& intuple, float val) {
            try {
                _traincount++;
                for (int i = 0; i < _nrams; i++) {
                    _rams[i][intuple[i]] = { _rams[i][intuple[i]].first + 1.0, _rams[i][intuple[i]].second + val } ;
                }
            } catch (const std::exception &e) {
                throw py::value_error(std::string("Wrong val arg: ") + e.what());
            }
        }
        
        float response_tpl_val(const std::vector<int>& intuple) {
            float count = 0.0;
            float acc_value = 0.0;
            for (int i = 0; i < _nrams; i++) {
                count += _rams[i][intuple[i]].first ;
                acc_value += _rams[i][intuple[i]].second;
            }
            if (count > 0)
                return acc_value / count;
            else 
                return 0.0;
        }   

        py::array_t<float> getMI() {
            std::vector<float> result(_retina_size, 0);
            int offset = 0;
    
            try {
                for (int neuron = 0; neuron < _nrams; ++neuron) {
                    const auto& ram = _rams.at(neuron);
                    for (int address = 0; address < _nloc; ++address) {
                        if (ram[address].first > 0) {
                            for (int b = 0; b < _nobits; ++b) {
                                if ((address >> (_nobits - 1 - b)) & 1) {
                                    int index = _mapping.at(offset + b);
                                    result.at(index) += ram[address].first;
                                    if (_maxvalue < result.at(index)) {
                                        _maxvalue = result.at(index);
                                    }
                                }
                            }
                        }
                    }
                    offset += _nobits;
                }
                return py::array_t<float>(result.size(), result.data());
            } catch (const std::exception &e) {
                throw py::value_error(std::string("Wrong y arg: ") + e.what());
            }
        }
    
        int getNRams() { return _nrams; }
        std::vector<int> getMapping() const { return _mapping;}
        int getNBits() const { return _nobits;}
        int getSize() const { return _retina_size;}
        int getTcount() const { return _traincount;}
        std::vector<std::vector<std::pair<float,float>>> getRams() const { return _rams;}
    };

class WiSARD {
private:
    int _nobits, _retina_size, _nrams, _nloc;
    std::vector<int> _mapping;
    std::vector<int> _revmapping;
    std::unordered_map<int, std::vector<std::vector<float>>> _layers;

    std::unordered_map<int, std::vector<float>> _mi;
    std::unordered_map<int, float> _maxvalue;
    std::unordered_map<int, int> _traincount;
    std::vector<int> _classes;


public:
    // Initialize mypowers (powers of 2)
    std::vector<int> mypowers = []() {
        std::vector<int> result(32);
        for (int i = 0; i < 32; ++i) {
            result[i] = static_cast<int>(std::pow(2, i));
        }
        return result;
    }();

    WiSARD(int size, int n_bits = 16, std::vector<int> classes = {0, 1}, int map = -1)
        : _nobits(n_bits), _retina_size(size), _classes(classes) {

        _nloc = 1 << _nobits;
        _nrams = (_retina_size % _nobits == 0) ? (_retina_size / _nobits) : (_retina_size / _nobits + 1);
        _mapping.resize(_retina_size);
        _revmapping.resize(_retina_size);

        for (int i = 0; i < _retina_size; ++i) {
            _mapping[i] = i;
        }

        if (map > -1) {
            std::mt19937 rng(map);  // Initialize random number generator with seed
            std::shuffle(_mapping.begin(), _mapping.end(), rng);  // Use std::shuffle for reproducibility
        
            for (int i = 0; i < _retina_size; ++i) {
                _revmapping[_mapping[i]] = i;
            }
        }

        for (int c : _classes) {
            _mi[c] = std::vector<float>(_retina_size, 0.0);
            _maxvalue[c] = 0;
            _traincount[c] = 0;
            _layers[c] = std::vector<std::vector<float>>(_nrams, std::vector<float>(_nloc, 0));
        }
    }

    std::vector<int> _mk_tuple(py::array_t<uint8_t> X) {
        auto buf = X.request();
        uint8_t* ptr = static_cast<uint8_t*>(buf.ptr);
        std::vector<int> intuple(_nrams, 0);
        
        for (int i = 0; i < _nrams; i++) {
            for (int j = 0; j < _nobits; j++) {
                int idx = _mapping[((i * _nobits) + j) % _retina_size];
                intuple[i] += (ptr[idx] > 0) ? (1 << (_nobits - 1 - j)) : 0;
            }
        }
        return intuple;
    }

    std::vector<int> _mk_tuple_float(py::array_t<float> X, int ntics, py::array_t<float> offsets, py::array_t<float> ranges) {
        auto X_buf = X.request();
        float* ptr = static_cast<float*>(X_buf.ptr);
        auto off_buf = offsets.request();
        float* off_ptr = static_cast<float*>(off_buf.ptr);
        auto rng_buf = ranges.request();
        float* rng_ptr = static_cast<float*>(rng_buf.ptr);
        std::vector<int> intuple(_nrams, 0);
        
        for (int i = 0; i < _nrams; i++) {
            for (int j = 0; j < _nobits; j++) {
                int x = _mapping[((i * _nobits) + j) % _retina_size];
                int index = x / ntics;
                int value = int((ptr[index] - off_ptr[index]) * ntics / rng_ptr[index]);
                intuple[i] += (x % ntics < value) ? (1 << (_nobits - 1 - j)) : 0;
            }
        }
        return intuple;
    }

    std::vector<int> _mk_tuple_img(py::array_t<int, py::array::c_style | py::array::forcecast> image, int h) {
        // Get buffer info from the NumPy array
        auto buf = image.request();

        // Ensure the input is a 2D array
        if (buf.ndim != 2) {
            throw std::runtime_error("Input image must be a 2D NumPy array");
        }

        int height = buf.shape[0];  // Number of rows
        int width = buf.shape[1];   // Number of columns
        int* ptr = static_cast<int*>(buf.ptr);  // Pointer to data

        std::vector<int> intuple(_nrams, 0);

        for (int i = 0; i < _nrams; ++i) {
            for (int j = 0; j < _nobits; ++j) {
                int x = _mapping[((i * _nobits) + j) % _retina_size];
                int J = x % h;  // Column index
                int I = x / h;  // Row index

                // Bounds checking to avoid out-of-bounds errors
                if (I < 0 || I >= height || J < 0 || J >= width) {
                    throw std::out_of_range("Index out of bounds in image lookup");
                }

                int base = mypowers[_nobits - 1 - j];
                intuple[i] += base * ptr[I * width + J];  // Access flattened NumPy array
            }
        }
        return intuple;
    }

    py::array_t<int> _mk_tuple_img_multi(py::array_t<int, py::array::c_style | py::array::forcecast> image, 
                                            int h, int dx = 1, int dy = 1, int res = 1) {
        // Get buffer info
        auto buf = image.request();

        // Ensure input is a 2D array
        if (buf.ndim != 2) {
            throw std::runtime_error("Input image must be a 2D NumPy array");
        }

        int width = buf.shape[1];   // Columns
        int height = buf.shape[0];  // Rows
        int* ptr = static_cast<int*>(buf.ptr);  // Pointer to image data

        // Create the output tensor: shape (2*dy+1, 2*dx+1, _nrams)
        std::vector<int> tmat((2 * dy + 1) * (2 * dx + 1) * _nrams, 0);

        int offx = dx * res;
        int offy = dy * res;

        for (int i = 0; i < _nrams; ++i) {
            for (int j = 0; j < _nobits; ++j) {
                int x = _mapping[((i * _nobits) + j) % _retina_size];
                int J = offy + (x % h);  // Column index
                int I = offx + (x / h);  // Row index

                // Ensure indices are within bounds
                if (I < 0 || I >= height || J < 0 || J >= width) continue;

                int base = mypowers[_nobits - 1 - j];

                // Get flat index in 3D array (dy, dx, i)
                auto index = [&](int y, int x) {
                    return (y * (2 * dx + 1) + x) * _nrams + i;
                };

                // Center pixel
                tmat[index(dy, dx)] += base * ptr[I * width + J];

                // Vertical shift
                int incry = res;
                for (int stepy = 1; stepy <= dy; ++stepy) {
                    if (J + incry < width)  tmat[index(stepy + dy, dx)] += base * ptr[I * width + (J + incry)];
                    if (J - incry >= 0)     tmat[index(-stepy + dy, dx)] += base * ptr[I * width + (J - incry)];

                    int incrx = res;
                    for (int stepx = 1; stepx <= dx; ++stepx) {
                        if (I + incrx < height) {
                            if (J + incry < width)  tmat[index(stepy + dy, stepx + dx)] += base * ptr[(I + incrx) * width + (J + incry)];
                            if (J - incry >= 0)     tmat[index(-stepy + dy, stepx + dx)] += base * ptr[(I + incrx) * width + (J - incry)];
                        }
                        if (I - incrx >= 0) {
                            if (J + incry < width)  tmat[index(stepy + dy, -stepx + dx)] += base * ptr[(I - incrx) * width + (J + incry)];
                            if (J - incry >= 0)     tmat[index(-stepy + dy, -stepx + dx)] += base * ptr[(I - incrx) * width + (J - incry)];
                        }
                        incrx += res;
                    }
                    incry += res;
                }

                // Horizontal shift
                int incrx = res;
                for (int stepx = 1; stepx <= dx; ++stepx) {
                    if (I + incrx < height) tmat[index(dy, stepx + dx)] += base * ptr[(I + incrx) * width + J];
                    if (I - incrx >= 0)     tmat[index(dy, -stepx + dx)] += base * ptr[(I - incrx) * width + J];
                    incrx += res;
                }
            }
        }

        // Convert std::vector to py::array_t<int> with correct shape
        return py::array_t<int>(
            {2 * dy + 1, 2 * dx + 1, _nrams}, // Shape
            tmat.data() // Data pointer
        );
    }
    

    void reinit() {
        for (int c : _classes) {
            std::fill(_mi[c].begin(), _mi[c].end(), 0);
            _maxvalue[c] = 0;
            _traincount[c] = 0;
            std::fill(_layers[c].begin(), _layers[c].end(), std::vector<float>(_nloc, 0));
        }
    }

    void train_tpl_val(const std::vector<int>& intuple, int y, float val) {
        try {
            _traincount.at(y)++;
            for (int i = 0; i < _nrams; i++) {
                _layers.at(y)[i][intuple[i]] += val;
            }
        } catch (const std::exception &e) {
            throw py::value_error(std::string("Wrong y or val args: ") + e.what());
        }
    }

    void train_tpl(const std::vector<int>& intuple, int y) {
        try {
            _traincount.at(y)++;
            for (int i = 0; i < _nrams; i++) {
                _layers.at(y)[i][intuple[i]] += 1.0f;
            }
        } catch (const std::exception &e) {
            throw py::value_error(std::string("Wrong y arg: ") + e.what());
        }
    }

    void train(py::array_t<uint8_t> X, int y) {
        try {
            std::vector<int> intuple = _mk_tuple(X);
            _traincount.at(y)++;
            for (int i = 0; i < _nrams; i++) {
                _layers.at(y)[i][intuple[i]] += 1.0f;
            }
        } catch (const std::exception &e) {
            throw py::value_error(std::string("Wrong y arg: ") + e.what());
        }
    }

    void trainforget(const py::array_t<int>& X, int y, float incr, float decr) {
        try {
            std::vector<int> intuple = _mk_tuple(X);
            _traincount[y]++;
            for (int i = 0; i < _nrams; ++i) {
                _layers[y][i][intuple[i]] += incr;
                for (int j = 0; j < intuple[i]; ++j) {
                    _layers[y][i][j] = std::max(0.0f, _layers[y][i][j] - decr);
                }
                for (int j = intuple[i] + 1; j < _nloc; ++j) {
                    _layers[y][i][j] = std::max(0.0f, _layers[y][i][j] - decr);
                }
            }
        } catch (const std::exception &e) {
            throw py::value_error(std::string("Wrong y arg: ") + e.what());
        }
    }

    std::unordered_map<int, float> response_tpl_val(const std::vector<int>& intuple) {
        std::unordered_map<int, float> results;

        for (auto& layer : _layers) {
            int y = layer.first;
            float accumulate = 0.0;
            for (int i = 0; i < _nrams; i++) {
                accumulate += layer.second[i][intuple[i]] ;
            }
            results[y] = static_cast<float>(accumulate) / _nrams;
        }
        return results;
    }

    std::unordered_map<int, float> response_tpl(const std::vector<int>& intuple, float threshold = 0.0, bool percentage = true) {
        std::unordered_map<int, float> results;

        for (auto& layer : _layers) {
            int y = layer.first;
            int count = 0;
            for (int i = 0; i < _nrams; i++) {
                if (layer.second[i][intuple[i]] > threshold) count++;
            }
            results[y] = static_cast<float>(count) / _nrams;
        }
        return results;
    }

    std::unordered_map<int, float> response(py::array_t<uint8_t> X, float threshold = 0.0, bool percentage = true) {
        std::vector<int> intuple = _mk_tuple(X);
        std::unordered_map<int, float> results;

        for (auto& layer : _layers) {
            int y = layer.first;
            int count = 0;
            for (int i = 0; i < _nrams; i++) {
                if (layer.second[i][intuple[i]] > threshold) count++;
            }
            results[y] = static_cast<float>(count) / _nrams;
        }
        return results;
    }

    int test(py::array_t<uint8_t> X, float threshold = 0.0) {
        auto responses = response(X, threshold);
        return std::max_element(responses.begin(), responses.end(),
                                [](const auto& a, const auto& b) { return a.second < b.second; })->first;
    }

    int test_tpl(const std::vector<int>& intuple, float threshold = 0.0) {
        auto responses = response_tpl(intuple, threshold);
        return std::max_element(responses.begin(), responses.end(),
                                [](const auto& a, const auto& b) { return a.second < b.second; })->first;
    }

    py::array_t<float> getMI(int y) {
        std::vector<float> result(_retina_size, 0);
        int offset = 0;

        try {
            for (int neuron = 0; neuron < _nrams; ++neuron) {
                const auto& ram = _layers.at(y).at(neuron);
                for (int address = 0; address < _nloc; ++address) {
                    if (ram[address] > 0) {
                        for (int b = 0; b < _nobits; ++b) {
                            if ((address >> (_nobits - 1 - b)) & 1) {
                                int index = _mapping.at(offset + b);
                                result.at(index) += ram[address];
                                if (_maxvalue.at(y) < result.at(index)) {
                                    _maxvalue.at(y) = result.at(index);
                                }
                            }
                        }
                    }
                }
                offset += _nobits;
            }
            return py::array_t<float>(result.size(), result.data());
        } catch (const std::exception &e) {
            throw py::value_error(std::string("Wrong y arg: ") + e.what());
        }
    }

    int getNRams() { return _nrams; }
    std::vector<int> getClasses() const { return _classes;}
    std::vector<int> getMapping() const { return _mapping;}
    int getNBits() const { return _nobits;}
    int getSize() const { return _retina_size;}
    const std::unordered_map<int, int> getTcounts() const { return _traincount;}

};

PYBIND11_MODULE(wisard, m) {
    py::class_<WiSARD>(m, "WiSARD")
        .def(py::init<int, int, std::vector<int>, int>(), py::arg("size"), py::arg("n_bits") = 16, py::arg("classes"), py::arg("map") = -1)
        .def("_mk_tuple", &WiSARD::_mk_tuple, "Make-tuple function", py::arg("X"))
        .def("_mk_tuple_float", &WiSARD::_mk_tuple_float, "Make-tuple function for floats", py::arg("X"), py::arg("ntics"), py::arg("offsets"),py::arg("ranges"))
        .def("_mk_tuple_img", &WiSARD::_mk_tuple_img, "Make-tuple function for image", py::arg("image"), py::arg("h"))
        .def("_mk_tuple_img_multi", &WiSARD::_mk_tuple_img_multi, "Make-tuple function for image", py::arg("image"), py::arg("h"), py::arg("dx") = 1, py::arg("dy") = 1, py::arg("res") = 1)
        .def("reinit", &WiSARD::reinit, "Initialization function")
        .def("train", &WiSARD::train, "Training function", py::arg("X"), py::arg("y"))
        .def("train_tpl", &WiSARD::train_tpl, "Training function with tuple input", py::arg("X"), py::arg("y"))
        .def("train_tpl_val", &WiSARD::train_tpl_val, "Training function with tuple input and value (for regression)", py::arg("X"), py::arg("y"), py::arg("val"))
        .def("trainforget", &WiSARD::trainforget, "Training/forgetting function", py::arg("X"), py::arg("y"), py::arg("incr"), py::arg("decr"))
        .def("response_tpl_val", &WiSARD::response_tpl_val, "Probability prediction function with tuple input (for regression)", py::arg("intuple"))
        .def("response_tpl", &WiSARD::response_tpl, "Probability prediction function with tuple input", py::arg("intuple"), py::arg("threshold")=0.0, py::arg("percentage")=true)
        .def("response", &WiSARD::response, "Probability prediction function", py::arg("X"), py::arg("threshold")=0.0, py::arg("percentage")=true)
        .def("test", &WiSARD::test, "Classification function", py::arg("X"), py::arg("threshold")=0.0)
        .def("test_tpl", &WiSARD::test_tpl, "Classification function with tuple input", py::arg("X"), py::arg("threshold")=0.0)
        .def("getMI", &WiSARD::getMI, "Mental Image getter function", py::arg("y"))
        .def("getNRams", &WiSARD::getNRams, "Number of Rams getter function")
        .def("getClasses", &WiSARD::getClasses, "Class list getter function")
        .def("getMapping", &WiSARD::getMapping, "Class mapping getter function")
        .def("getNBits", &WiSARD::getNBits, "Number of bits getter function")
        .def("getSize", &WiSARD::getSize, "Retina size getter function")
        .def("getTcounts", &WiSARD::getTcounts, "Training count getter function");
    py::class_<WiSARDreg>(m, "WiSARDreg")
        .def(py::init<int, int, int>(), py::arg("size"), py::arg("n_bits") = 16, py::arg("map") = -1)
        .def("_mk_tuple_float", &WiSARDreg::_mk_tuple_float, "Make-tuple function for floats", py::arg("X"), py::arg("ntics"), py::arg("offsets"),py::arg("ranges"))
        .def("train_tpl_val", &WiSARDreg::train_tpl_val, "Training function with tuple input and value (for regression)", py::arg("intuple"), py::arg("val"))
        .def("response_tpl_val", &WiSARDreg::response_tpl_val, "Probability prediction function with tuple input (for regression)", py::arg("intuple"))
        .def("getMI", &WiSARDreg::getMI, "Mental Image getter function")
        .def("getNRams", &WiSARDreg::getNRams, "Number of Rams getter function")
        .def("getMapping", &WiSARDreg::getMapping, "Class mapping getter function")
        .def("getNBits", &WiSARDreg::getNBits, "Number of bits getter function")
        .def("getSize", &WiSARDreg::getSize, "Retina size getter function")
        .def("getTcount", &WiSARDreg::getTcount, "Training count getter function")
        .def("getRams", &WiSARDreg::getRams, "RAM getter function");
   }

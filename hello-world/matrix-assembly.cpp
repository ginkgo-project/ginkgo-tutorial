#include <ginkgo/ginkgo.hpp>

int main() {
    using Mtx = gko::matrix::Csr<double, int>;
    using Vec = gko::matrix::Dense<double>;
    using dim = gko::dim<2>;
    using matrix_data = gko::matrix_data<double, int>;
    auto exec = gko::ReferenceExecutor::create();
    matrix_data data{dim{10, 10}};
    auto mtx = Mtx::create(exec);
    auto x = Vec::create(exec, dim{10, 1});
    auto y = Vec::create(exec, dim{10, 1});
    for (int row = 0; row < 10; row++) {
        data.nonzeros.emplace_back(row, row, 2.0);
        data.nonzeros.emplace_back(row, (row + 1) % 10, -1.0);
        data.nonzeros.emplace_back(row, (row + 9) % 10, -1.0);
        x->at(row, 0) = 1.0; // only works on CPU executors
    }
    mtx->read(data);
    mtx->apply(x, y);
    gko::write(std::cout, y);
}

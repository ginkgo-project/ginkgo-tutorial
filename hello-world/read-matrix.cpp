#include <ginkgo/ginkgo.hpp>
#include <fstream>

int main() {
    using Mtx = gko::matrix::Csr<double, int>;
    using Vec = gko::matrix::Dense<double>;
    auto exec = gko::ReferenceExecutor::create();
    auto mtx = gko::read<Mtx>(std::ifstream{"data/A.mtx"}, exec);
    auto x = gko::read<Vec>(std::ifstream{"data/x.mtx"}, exec);
    auto y = gko::read<Vec>(std::ifstream{"data/y.mtx"}, exec);
    mtx->apply(x, y);
    gko::write(std::cout, y);
}
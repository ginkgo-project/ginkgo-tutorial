#include <ginkgo/ginkgo.hpp>

int main() {
    using Mtx = gko::matrix::Csr<double, int>;
    using Vec = gko::matrix::Dense<double>;
    auto exec = gko::ReferenceExecutor::create();
    auto mtx = gko::initialize<Mtx>({{0.0, 1.0}, {1.0, 0.0}}, exec);
    auto vec = gko::initialize<Vec>({2.0, 3.0}, exec);
    auto result = vec->clone();
    mtx->apply(vec, result);
    gko::write(std::cout, result);
}

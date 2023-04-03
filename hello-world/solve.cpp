#include <ginkgo/ginkgo.hpp>

int main()
{
    using Mtx = gko::matrix::Csr<double, int>;
    using Vec = gko::matrix::Dense<double>;
    using Solver = gko::solver::Cg<double>;
    using NormCriterion = gko::stop::ResidualNorm<double>;
    using IterationCriterion = gko::stop::Iteration;
    using matrix_data = gko::matrix_data<double, int>;
    auto exec = gko::ReferenceExecutor::create();
    gko::size_type size = 100;
    matrix_data mtx_data{gko::dim<2>{size,size}};
    auto mtx = gko::share(Mtx::create(exec));
    auto x = Vec::create(exec, gko::dim<2>{size, 1});
    auto y = Vec::create(exec, gko::dim<2>{size, 1});
    for (int row = 0; row < size; row++) {
        mtx_data.nonzeros.emplace_back(row, row, 4.0);
        mtx_data.nonzeros.emplace_back(row, (row + 1) % size, 1.0);
        mtx_data.nonzeros.emplace_back(row, (row + size - 1) % size, 1.0);
        x->at(row, 0) = row;
        y->get_values()[row] = 0.0;
    }
    mtx->read(mtx_data);
    auto norm_criterion = gko::share(NormCriterion::build()
                                             .with_baseline(gko::stop::mode::rhs_norm)
                                             .with_reduction_factor(1e-10)
                                             .on(exec));
    auto it_criterion = gko::share(IterationCriterion::build()
                                           .with_max_iters(100)
                                           .on(exec));
    auto solver_factory = Solver::build()
            .with_criteria(it_criterion, norm_criterion)
            .with_preconditioner(
                    gko::preconditioner::Jacobi<>::build()
                            .with_max_block_size(4).on(exec))
            .on(exec);
    auto solver = solver_factory->generate(mtx);
    solver->apply(x, y);
    auto one = gko::initialize<Vec>({1.0}, exec);
    auto neg_one = gko::initialize<Vec>({-1.0}, exec);
    auto res = gko::initialize<Vec>({0.0}, exec);
    mtx->apply(one, y, neg_one, x);
    x->compute_norm2(res);
    gko::write(std::cout, res);
}
"""Render every figure of the manuscript and supplement."""
from src.figures import fig_concepts, fig_problem_a, fig_problem_b, fig_problem_c, fig_problem_d, fig_references

if __name__ == "__main__":
    fig_concepts.main()
    fig_problem_a.main()
    fig_problem_b.main()
    fig_problem_c.main()
    fig_problem_c.kappa_figure()
    fig_problem_d.main()
    fig_references.main()

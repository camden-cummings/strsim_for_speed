import lazy_loader


__getattr__, __dir__, __all__ = lazy_loader.attach(
    __name__,
    submodules={
        'comparison',
        'helpers',
        'speedy_str_sim',
        'structural_sim_from_scratch',
    },
    submod_attrs={
        'comparison': [
            'correlate1d_x',
            'correlate1d_y',
        ],
        'helpers': [
            'calc_mode',
            'calc_mode_img',
            'get_contour_mask',
        ],
        'speedy_str_sim': [
            'check_similarity',
            'run_correlate_rearr_y',
            'run_mode_rearr_y',
            'update_corr_rearr_y',
        ],
        'structural_sim_from_scratch': [
            'correlate1d_x',
            'correlate1d_y',
            'correlate1d_y_wrap',
            'generate_weights',
            'normalize_diff',
            'run_math',
            'run_math_complete',
            'setup',
        ],
    },
)

__all__ = ['calc_mode', 'calc_mode_img', 'check_similarity', 'comparison',
           'correlate1d_x', 'correlate1d_y', 'correlate1d_y_wrap',
           'generate_weights', 'get_contour_mask', 'helpers', 'normalize_diff',
           'run_correlate_rearr_y', 'run_math', 'run_math_complete',
           'run_mode_rearr_y', 'setup', 'speedy_str_sim',
           'structural_sim_from_scratch', 'update_corr_rearr_y']


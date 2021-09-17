from pathlib import Path

import voluptuous as vo


schema = vo.Schema(
    vo.All(
        {
            vo.Required("version"): vo.In([1.0]),
            vo.Required("input_dir"): vo.Coerce(Path),
            vo.Required("output_dir"): vo.Coerce(Path),
            vo.Required("analysis_parameters"): vo.Coerce(Path),
            vo.Required("data_organization"): vo.Coerce(Path),
            vo.Required("codebook"): vo.Coerce(Path),
            vo.Required("microscope_parameters"): vo.Coerce(Path),
            vo.Required("positions"): vo.Coerce(Path),
        },
    )
)

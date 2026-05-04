import pyfar as pf
import sofar as sf
import scipy as sc
from importlib import import_module
import tempfile
from head_orientation_class import HeadOrientations
import os


_MATLAB_ENGINE = None


def _get_matlab_engine():
    global _MATLAB_ENGINE

    if _MATLAB_ENGINE is None:
        matlab_engine = import_module("matlab.engine")
        shared_sessions = matlab_engine.find_matlab()

        if shared_sessions:
            _MATLAB_ENGINE = matlab_engine.connect_matlab(shared_sessions[0])
        else:
            _MATLAB_ENGINE = matlab_engine.start_matlab()

        _MATLAB_ENGINE.amt_start(nargout=0)
        _MATLAB_ENGINE.SOFAstart(nargout=0)

    return _MATLAB_ENGINE

def _get_subset(sofa, sampling):
    eng = _get_matlab_engine()
    idx = eng.SOFAfind(sofa, sampling.azimuth,
                       sampling.elevation, nargout=1)

    eng.workspace['sofa'] = sofa
    eng.workspace['idx'] = idx

    eng.eval("new_sofa=sofa;", nargout=0)
    eng.eval("new_sofa.Data.IR = sofa.Data.IR(idx,:,:);", nargout=0)
    eng.eval("new_sofa.SourcePosition = sofa.SourcePosition(idx,:,:);", nargout=0)

    sofa_subset = eng.eval("new_sofa", nargout=1)
    return sofa_subset


def barumerli_localization(
    template_head_orientations: HeadOrientations,
    target_head_orientations: HeadOrientations,
    subsampling: pf.Coordinates = None,
    output_dir: str = None,
    repetitions: int = 100,
    save_matrix: bool = False,
):
    """"""
    eng = _get_matlab_engine()

    if target_head_orientations.n_orientations != 1 \
        and target_head_orientations.n_orientations != \
            template_head_orientations.n_orientations:
        raise ValueError("Don't do this")

    # extract target_features
    sofa_target = eng.SOFAload(
        str(target_head_orientations.sofa_file_paths[0]), nargout=1)

    sofa_target = _get_subset(sofa_target, subsampling)

    feat_target = eng.barumerli2023_NOINTERPOLATION_featureextraction(
        sofa_target,
        'target',
        'pge',
        nargout=1)

    results = []

    for head_orientation in template_head_orientations:
        # If the template has beed interpolated before, there is no
        # corresponding sofa file, so we create a temporary one
        if head_orientation.sofa_file_paths is None:
            print("creating tempdir for head template")
            sofa = sf.Sofa("SimpleFreeFieldHRIR")
            sofa.SourcePosition = head_orientation.source_positions.spherical_elevation

            sofa.Data_IR = head_orientation.hrirs.time[0]
            sofa.Data_SamplingRate = head_orientation.hrirs.sampling_rate

            with tempfile.TemporaryDirectory() as tmpdir:
                file_path = f"{tmpdir}/example.sofa"

                # Save the SOFA file
                sf.write_sofa(file_path, sofa)

                # You can read it back if needed
                sofa_template = eng.SOFAload(file_path, nargout=1)

        else:
            # exctract template features for current ho
            sofa_template = \
                eng.SOFAload(str(template_head_orientations.sofa_file_paths[0]),
                            nargout=1)

        sofa_template = _get_subset(sofa_template, subsampling)

        feat_template = \
            eng.barumerli2023_NOINTERPOLATION_featureextraction(sofa_template,
                                                                'template',
                                                                'pge',
                                                                nargout=1)

        # get prediction
        prediction_matrix = eng.barumerli2023('template', feat_template,
                                              'target', feat_target,
                                              'num_exp', repetitions)

        metrics = eng.barumerli2023_metrics(prediction_matrix,
                                            'middle_metrics')

        if output_dir:
            orientation = head_orientation.head_orientations
            filename = f"metrics_bend_{int(orientation[:, 0])}" \
                f"elev_{int(orientation[:, 1])}" \
                    f"azim{int(orientation[:, 2])}.mat"
            filepath = os.path.join(output_dir, filename)
            sc.io.savemat(filepath, metrics)
            print(f"saved to {filepath}")

        results.append(metrics)

    return results, template_head_orientations.head_orientations

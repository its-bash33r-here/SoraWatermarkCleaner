"""
Basic test to verify test setup works without external dependencies.
"""

import sys
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_schemas_import():
    """
    Verify that CleanerType constants import correctly and have expected values.
    
    Checks that CleanerType.LAMA equals "lama" and CleanerType.E2FGVI_HQ equals "e2fgvi_hq".
    
    Returns:
        bool: True if both checks pass, False otherwise.
    """
    try:
        from sorawm.schemas import CleanerType

        assert CleanerType.LAMA == "lama"
        assert CleanerType.E2FGVI_HQ == "e2fgvi_hq"
        print("✓ Schemas import test passed")
    except Exception as e:
        print(f"✗ Schemas import test failed: {e}")
        return False
    return True


def test_watermark_cleaner_factory():
    """
    Verify WaterMarkCleaner returns the correct cleaner instances for supported CleanerType values using mocks.
    
    Tests that requesting CleanerType.LAMA yields the LamaCleaner instance and requesting CleanerType.E2FGVI_HQ yields the E2FGVIHDCleaner instance by patching those cleaner classes.
    
    Returns:
        result (bool): `True` if both cleaner selections produced the expected mocked instances, `False` otherwise.
    """
    try:
        from unittest.mock import patch, MagicMock

        with (
            patch("sorawm.watermark_cleaner.LamaCleaner") as mock_lama,
            patch("sorawm.watermark_cleaner.E2FGVIHDCleaner") as mock_e2fgvi,
        ):
            mock_lama.return_value = "lama_instance"
            mock_e2fgvi.return_value = "e2fgvi_instance"

            from sorawm.schemas import CleanerType
            from sorawm.watermark_cleaner import WaterMarkCleaner

            # Test LAMA cleaner
            cleaner = WaterMarkCleaner(CleanerType.LAMA)
            assert cleaner == "lama_instance"
            mock_lama.assert_called_once()

            # Test E2FGVI cleaner
            cleaner = WaterMarkCleaner(CleanerType.E2FGVI_HQ)
            assert cleaner == "e2fgvi_instance"
            mock_e2fgvi.assert_called_once()

        print("✓ WaterMarkCleaner factory test passed")
    except Exception as e:
        print(f"✗ WaterMarkCleaner factory test failed: {e}")
        return False
    return True


def test_imputation_utils():
    """
    Run basic tests for imputation utilities (breakpoint detection and interval bbox averaging) using mocks.
    
    Mocks ruptures' API to simulate a change-point detection result and verifies that find_2d_data_bkps returns the expected breakpoints list (excluding the final point). Also verifies that get_interval_average_bbox computes an averaged bounding box for intervals containing valid boxes. Intended for lightweight verification without external dependencies.
    
    Returns:
        bool: `True` if all assertions pass, `False` if an exception occurs during the tests.
    """
    try:
        # Test with minimal data that doesn't require ruptures
        import numpy as np

        # Mock the imputation functions to avoid ruptures dependency
        from unittest.mock import patch

        with patch("sorawm.utils.imputation_utils.rpt") as mock_rpt:
            # Mock the CPD result
            mock_algo = MagicMock()
            mock_algo.predict.return_value = [5, 10]
            mock_rpt.KernelCPD.return_value.fit.return_value = mock_algo

            from sorawm.utils.imputation_utils import find_2d_data_bkps

            data = [(1, 2), (3, 4), (5, 6)]
            result = find_2d_data_bkps(data)

            assert result == [5]  # Should return bkps[:-1]

        # Test bbox averaging function
        from sorawm.utils.imputation_utils import get_interval_average_bbox

        bboxes = [(10, 20, 30, 40), (11, 21, 31, 41), None]
        bkps = [0, 3]
        result = get_interval_average_bbox(bboxes, bkps)

        assert len(result) == 1
        assert result[0] is not None  # Should average the two valid bboxes

        print("✓ Imputation utils test passed")
    except Exception as e:
        print(f"✗ Imputation utils test failed: {e}")
        return False
    return True


def test_video_utils():
    """
    Run a basic smoke test for the video utilities' frame merging behavior.
    
    Validates that merge_frames_with_overlap correctly returns the two provided RGB frames when merging a chunk with zero overlap.
    
    Returns:
        True if the test passed, False otherwise.
    """
    try:
        from sorawm.utils.video_utils import merge_frames_with_overlap
        import numpy as np

        # Test basic merging
        frame1 = np.ones((10, 10, 3), dtype=np.uint8) * 100
        frame2 = np.ones((10, 10, 3), dtype=np.uint8) * 200

        result = merge_frames_with_overlap(
            result_frames=None,
            chunk_frames=[frame1, frame2],
            start_idx=0,
            overlap_size=0,
            is_first_chunk=True,
        )

        assert len(result) == 2
        assert np.array_equal(result[0], frame1)
        assert np.array_equal(result[1], frame2)

        print("✓ Video utils test passed")
    except Exception as e:
        print(f"✗ Video utils test failed: {e}")
        return False
    return True


if __name__ == "__main__":
    print("Running basic tests...")
    print("=" * 50)

    tests = [
        test_schemas_import,
        test_watermark_cleaner_factory,
        test_imputation_utils,
        test_video_utils,
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        if test():
            passed += 1
        print()

    print("=" * 50)
    print(f"Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All basic tests passed!")
        sys.exit(0)
    else:
        print("❌ Some tests failed")
        sys.exit(1)
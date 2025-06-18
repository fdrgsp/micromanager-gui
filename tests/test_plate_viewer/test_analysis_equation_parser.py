"""Test equation parsing functionality in the analysis module."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np

from micromanager_gui._plate_viewer._util import equation_from_str


@patch("micromanager_gui._plate_viewer._analysis.show_error_dialog")
def test_equation_parser_comprehensive(mock_error_dialog, qapp):
    """Comprehensive test for equation parsing functionality."""
    # ===== VALID EQUATIONS TESTS =====

    # Standard linear equation
    func = equation_from_str("y = 2*x + 3")
    assert func is not None
    assert func(5) == 13  # 2*5 + 3 = 13

    # Linear equation with decimal coefficients
    func = equation_from_str("y = 11.07*x - 6.63")
    assert func is not None
    assert abs(func(1) - 4.44) < 0.01  # 11.07*1 - 6.63 = 4.44

    # Linear equation with negative slope
    func = equation_from_str("y = -0.5*x + 10")
    assert func is not None
    assert func(4) == 8  # -0.5*4 + 10 = 8

    # Linear equation with spaces
    func = equation_from_str("y = 2 * x + 3")
    assert func is not None
    assert func(5) == 13

    # Standard quadratic equation
    func = equation_from_str("y = 1*x^2 + 2*x + 1")
    assert func is not None
    assert func(3) == 16  # 1*3^2 + 2*3 + 1 = 9 + 6 + 1 = 16

    # Quadratic with decimal coefficients
    func = equation_from_str("y = 0.5*x^2 + 1*x + 2")
    assert func is not None
    assert func(2) == 6  # 0.5*4 + 2 + 2 = 2 + 2 + 2 = 6

    # Quadratic with spaces
    func = equation_from_str("y = 1 * x^2 + 2 * x + 1")
    assert func is not None
    assert func(3) == 16

    # Simple exponential
    func = equation_from_str("y = 2*exp(0*x) + 1")
    assert func is not None
    assert abs(func(5) - 3) < 0.01  # 2*exp(0) + 1 = 2*1 + 1 = 3

    # Exponential with positive exponent
    func = equation_from_str("y = 1*exp(0.1*x) + 0")
    assert func is not None
    result = func(0)
    assert abs(result - 1) < 0.01  # 1*exp(0) + 0 = 1

    # Exponential with spaces
    func = equation_from_str("y = 2 * exp(0 * x) + 1")
    assert func is not None
    assert abs(func(5) - 3) < 0.01

    # Square root
    func = equation_from_str("y = 2*x^0.5 + 1")
    assert func is not None
    assert func(4) == 5  # 2*sqrt(4) + 1 = 2*2 + 1 = 5

    # Square
    func = equation_from_str("y = 1*x^2 + 0")
    assert func is not None
    assert func(3) == 9  # 1*3^2 + 0 = 9

    # Cube
    func = equation_from_str("y = 0.5*x^3 + 1")
    assert func is not None
    assert func(2) == 5  # 0.5*8 + 1 = 4 + 1 = 5

    # Power with spaces
    func = equation_from_str("y = 2 * x^0.5 + 1")
    assert func is not None
    assert func(4) == 5

    # Natural logarithm
    func = equation_from_str("y = 1*log(x) + 0")
    assert func is not None
    result = func(np.e)
    assert abs(result - 1) < 0.01  # 1*log(e) + 0 = 1*1 + 0 = 1

    # Logarithm with coefficient
    func = equation_from_str("y = 2*log(x) + 1")
    assert func is not None
    result = func(np.e)
    assert abs(result - 3) < 0.01  # 2*log(e) + 1 = 2*1 + 1 = 3

    # Logarithm with spaces
    func = equation_from_str("y = 1 * log(x) + 0")
    assert func is not None
    result = func(np.e)
    assert abs(result - 1) < 0.01

    # Empty string input
    func = equation_from_str("")
    assert func is None

    # ===== INVALID EQUATIONS TESTS =====

    # Missing coefficient
    func = equation_from_str("y = x")
    assert func is None

    # Missing multiplication sign
    func = equation_from_str("y = 2x + 3")
    assert func is None

    # Completely invalid
    func = equation_from_str("invalid equation")
    assert func is None

    # Invalid format
    func = equation_from_str("x = 2*y + 3")
    assert func is None

    # Missing operator
    func = equation_from_str("y = 2 3")
    assert func is None

    # The equation_from_str function logs errors instead of showing dialogs
    # so we don't check for error dialog calls

    # ===== CASE INSENSITIVE TESTS =====

    # Uppercase
    func = equation_from_str("Y = 2*X + 3")
    assert func is not None
    assert func(5) == 13

    # Mixed case
    func = equation_from_str("y = 2*X + 3")
    assert func is not None
    assert func(5) == 13

    # Exponential with uppercase
    func = equation_from_str("Y = 2*EXP(0*X) + 1")
    assert func is not None
    assert abs(func(5) - 3) < 0.01

    # Logarithm with uppercase
    func = equation_from_str("Y = 1*LOG(X) + 0")
    assert func is not None
    result = func(np.e)
    assert abs(result - 1) < 0.01

    # ===== EDGE CASES TESTS =====

    # Zero coefficients
    func = equation_from_str("y = 0*x + 5")
    assert func is not None
    assert func(100) == 5  # Should always return 5

    # Very small numbers
    func = equation_from_str("y = 0.001*x + 0.002")
    assert func is not None
    assert abs(func(1000) - 1.002) < 0.001

    # Very large numbers
    func = equation_from_str("y = 1000*x + 2000")
    assert func is not None
    assert func(1) == 3000

    # ===== ARRAY INPUT TESTS =====

    func = equation_from_str("y = 2*x + 3")
    assert func is not None

    # Test with numpy array
    x_vals = np.array([1, 2, 3, 4, 5])
    expected = np.array([5, 7, 9, 11, 13])  # 2*x + 3
    result = func(x_vals)

    assert isinstance(result, np.ndarray)
    np.testing.assert_array_almost_equal(result, expected)

    # ===== SPECIFIC USE CASE TESTS =====

    # Default equation from the GUI
    func = equation_from_str("y = 11.07 * x - 6.63")
    assert func is not None
    assert abs(func(1) - 4.44) < 0.01

    # Common power curve
    func = equation_from_str("y = 0.1*x^2 + 0.5*x + 1")
    assert func is not None
    assert func(10) == 16  # 0.1*100 + 0.5*10 + 1 = 10 + 5 + 1 = 16

    # Exponential growth for high-power LEDs
    func = equation_from_str("y = 1*exp(0.05*x) + 0")
    assert func is not None
    result = func(0)
    assert abs(result - 1) < 0.01  # exp(0) = 1

    # Square root response (common in optical systems)
    func = equation_from_str("y = 5*x^0.5 + 2")
    assert func is not None
    assert func(16) == 22  # 5*4 + 2 = 22

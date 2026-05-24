def test_basic_ved_formula():
    """
    VED = P / (v * h * t)

    Example:
    P = 200 W
    v = 1000 mm/s
    h = 0.1 mm
    t = 0.03 mm

    VED = 200 / (1000 * 0.1 * 0.03)
        = 66.666...
    """
    laser_power_w = 200
    scan_speed_mm_s = 1000
    hatch_spacing_mm = 0.1
    layer_thickness_mm = 0.03

    ved = laser_power_w / (
        scan_speed_mm_s * hatch_spacing_mm * layer_thickness_mm
    )

    assert round(ved, 2) == 66.67
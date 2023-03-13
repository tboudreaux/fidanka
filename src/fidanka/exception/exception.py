def shape_dimension_check(X, Y, dim=1):
    if X.shape != Y.shape:
        raise RuntimeError(f"X ({X.shape}) and Y ({Y.shape}) "
                           "must have the same shape")
    if len(X.shape) != dim or len(Y.shape) != dim:
        raise RuntimeError(f"X ({X.shape}) and Y ({Y.shape}) must "
                           f"be have dimension {dim}, not {len(X.shape)})")

"""A small pipeline that runs named steps in order.

Transformers provide fit(X) and transform(X). Estimators provide fit(X, y)
and predict(X). Classes only need these methods; inheritance is not required.
Loops below iterate over steps, not individual samples or array elements.

Extended directly from numcompute.pipeline. The original batch methods are
retained, with partial_fit added for chunk-wise updates. Components must also
provide partial_fit when the pipeline is used incrementally.
"""

from typing import Protocol

import numpy as np

__all__ = ["Transformer", "Estimator", "Pipeline"]


class Transformer(Protocol):
    """Interface for a preprocessor: learn parameters, then transform data."""

    def fit(self, X): 
        """Learn preprocessing parameters from X.

        Parameters
        ----------
        X : array_like, shape (n_samples, n_features)
            Training features.

        Returns
        -------
        Transformer
            The fitted object (self).

        Raises
        ------
        Exception
            Input validation and fitted-state errors depend on the implementation.

        Notes
        -----
        Time and space depend on the concrete implementation. This protocol
        only declares an interface; it performs no computation.
        """
        ...

    def partial_fit(self, X):
        """Update preprocessing parameters from a new chunk.

        Parameters
        ----------
        X : array_like, shape (n_samples, n_features)
            New feature rows with the existing input schema.

        Returns
        -------
        Transformer
            The updated object (self).

        Raises
        ------
        Exception
            Validation and update errors depend on the implementation.

        Notes
        -----
        Time and space depend on the concrete transformer. This protocol only
        declares the interface; it performs no computation or transformation.
        """
        ...

    def transform(self, X): 
        """Apply learned preprocessing parameters without refitting.

        Parameters
        ----------
        X : array_like, shape (n_samples, n_features)
            Features compatible with the fitted transformer.

        Returns
        -------
        array_like, shape (n_samples, n_output_features)
            Processed features.

        Raises
        ------
        Exception
            Input validation and fitted-state errors depend on the implementation.

        Notes
        -----
        Time and space depend on the concrete implementation. This protocol
        only declares an interface; it performs no computation.
        """
        ...


class Estimator(Protocol):
    """Interface for a model; this module does not implement a model."""

    def fit(self, X, y): 
        """Learn estimator state from features and targets.

        Parameters
        ----------
        X : array_like, shape (n_samples, n_features)
            Training features.
        y : array_like, usually shape (n_samples,)
            Target values aligned with X.

        Returns
        -------
        Estimator
            The fitted object (self).

        Raises
        ------
        Exception
            Input validation and fitted-state errors depend on the implementation.

        Notes
        -----
        Time and space depend on the concrete implementation. This protocol
        only declares an interface; it performs no computation.
        """
        ...

    def partial_fit(self, X, y):
        """Update an estimator from one aligned feature/target chunk.

        Parameters
        ----------
        X : array_like, shape (n_samples, n_features)
            New training features.
        y : array_like, shape (n_samples,)
            Target labels aligned with X.

        Returns
        -------
        Estimator
            The updated object (self).

        Raises
        ------
        Exception
            Validation, buffering and training errors depend on the estimator.

        Notes
        -----
        Time and space depend on the concrete estimator. A buffered tree may
        collect samples without training yet. This protocol performs no work.
        """
        ...

    def predict(self, X): 
        """Return predictions from fitted estimator state.

        Parameters
        ----------
        X : array_like, shape (n_samples, n_features)
            New features compatible with the fitted estimator.

        Returns
        -------
        array_like, usually shape (n_samples,)
            Predictions; exact shape depends on the estimator.

        Raises
        ------
        Exception
            Input validation and fitted-state errors depend on the implementation.

        Notes
        -----
        Time and space depend on the concrete implementation. This protocol
        only declares an interface; it performs no computation.
        """
        ...


class Pipeline:
    """Chain named transformers and an optional final estimator.

    Parameters
    ----------
    steps : iterable of (str, object) pairs
        Nonempty sequence with unique nonempty names. Each component must
        provide fit. Intermediate components must provide transform; the
        final component must provide transform or predict.

    Attributes
    ----------
    steps : list of (str, object) pairs
        Components in execution order, stored by reference without cloning.
    is_fitted_ : bool
        Whether a fit or nonempty partial_fit completed successfully. A buffered
        estimator may still be waiting for its first actual training event.

    Raises
    ------
    ValueError
        If the step sequence, names or component methods are invalid.
    TypeError
        If the step sequence or entries cannot be iterated/unpacked.

    Notes
    -----
    Construction takes O(s) time and space for s steps. Method costs are the
    sum of component costs plus O(s) dispatch. Working space is determined
    by component state and intermediate arrays; results are not cached.
    Batch-only components are accepted for the original fit/transform/predict
    API. partial_fit checks all update methods before changing any component.
    Updates use each preprocessor's current parameters. Previously transformed
    model buffers are not reprocessed, and this class does not expand a model's
    feature schema. Use stable preprocessing with buffered models; changing
    scaling/encoding can change feature coordinates or column meanings.
    """

    def __init__(self, steps):
        """Store and validate the named processing steps.

        Parameters
        ----------
        steps : iterable of (str, object) pairs
            Nonempty sequence of components with unique nonempty names. Every
            step requires fit; intermediate steps also require transform. The
            last step must provide transform or predict.

        Returns
        -------
        None
            Initialize an unfitted pipeline with the supplied component objects.

        Raises
        ------
        ValueError
            If steps are empty, names are invalid or required methods are absent;
            also raised if a step cannot be unpacked into a pair.
        TypeError
            If steps is not iterable or an entry cannot be unpacked.

        Notes
        -----
        Time and space: O(s), where s is the number of steps. Components are
        stored by reference and fitted in place, without cloning.
        """
        self.steps = list(steps)
        if not self.steps:
            raise ValueError("Pipeline must contain at least one step")

        names = set()
        for name, step in self.steps:
            if not isinstance(name, str) or not name or name in names:
                raise ValueError("Step names must be unique nonempty strings")
            if not callable(getattr(step, "fit", None)):
                raise ValueError(f"Step '{name}' must provide fit")
            
            names.add(name)

        for name, step in self.steps[:-1]:
            if not callable(getattr(step, "transform", None)):
                raise ValueError(f"Intermediate step '{name}' must provide transform")

        last = self.steps[-1][1]
        if not callable(getattr(last, "transform", None)) and not callable(getattr(last, "predict", None)):
            raise ValueError("The last step must provide transform or predict")
        
        self.is_fitted_ = False

    def fit(self, X, y=None):
        """Fit each step in order and return the pipeline.

        Parameters
        ----------
        X : array_like, shape (n_samples, n_features)
            Features accepted by the first step. Each later step receives the
            preceding transformer's output.
        y : array_like, usually shape (n_samples,), or None, default None
            Targets passed unchanged to the final step only if y is supplied
            and that step provides predict. Preprocessors receive X only.

        Returns
        -------
        Pipeline
            The fitted instance (self).

        Raises
        ------
        Exception
            Validation and computation errors raised by individual steps propagate.
        TypeError
            May arise if an estimator requires y but none is supplied.

        Notes
        -----
        Time: O(s) dispatch plus all fits and intermediate transforms. Space:
        component state and intermediate arrays; no transformed data is cached.
        Each intermediate step is fitted and then transforms its input. The
        last step is only fitted. A failed fit leaves is_fitted_ false.
        """
        self.is_fitted_ = False
        X_out = X
        for name, step in self.steps[:-1]:
            step.fit(X_out)
            X_out = step.transform(X_out)

        last = self.steps[-1][1]
        if y is not None and callable(getattr(last, "predict", None)):
            last.fit(X_out, y)
        else:
            last.fit(X_out)

        self.is_fitted_ = True
        return self

    def partial_fit(self, X, y=None):
        """Update each step in the same order as the original fit method.

        Parameters
        ----------
        X : array_like, shape (n_samples, n_features)
            New rows accepted by the first step. Intermediate transformers
            update and transform this chunk before passing it to the next step.
        y : array_like, shape (n_samples,), or None, default None
            Targets passed unchanged to the final step only when supplied and
            that step provides predict. Preprocessors receive features only.

        Returns
        -------
        Pipeline
            self. Empty, shape-compatible chunks do nothing and do not mark an
            uninitialised pipeline as fitted.

        Raises
        ------
        ValueError
            If a step lacks partial_fit, X is not 2-D/has no features, the first
            step's learned feature count changes, or y is not an aligned vector.
        TypeError
            May arise if the final estimator requires y but none is supplied.
        Exception
            Component validation/update errors propagate. If an update fails
            after execution starts, is_fitted_ is false; earlier component
            updates are not rolled back, just as with the original fit.

        Notes
        -----
        Time: O(s) dispatch plus all updates and intermediate transforms.
        Validation/conversion can use O(n*d+n) space; further space follows
        component states/outputs. No historical rows or outputs are cached by
        the pipeline, and fit is never substituted for a missing partial_fit.

        This method follows fit's call chain, replacing step.fit with
        step.partial_fit. Transform-only steps can change their output width.
        Downstream components must support that change; errors are propagated.
        A successful buffered-tree update need not make predict available until
        its buffer fills. Use stable preprocessing so its stored chunks and
        later predictions share the same feature representation.
        """
        for name, step in self.steps:
            if not callable(getattr(step, "partial_fit", None)):
                raise ValueError(f"Step '{name}' must provide partial_fit for streaming")

        X = np.asarray(X)
        if X.ndim != 2 or X.shape[1] == 0:
            raise ValueError("X must be a 2-D array (samples, features) with at least one feature")
        expected = getattr(self.steps[0][1], "n_features_in_", None)
        if expected is not None and X.shape[1] != expected:
            raise ValueError(f"expected {expected} features, got {X.shape[1]}")
        if y is not None:
            y = np.asarray(y)
            if y.ndim != 1 or len(y) != len(X):
                raise ValueError("y must be a 1-D array with one target per sample")
        if len(X) == 0:
            return self

        self.is_fitted_ = False
        X_out = X
        for name, step in self.steps[:-1]:
            step.partial_fit(X_out)
            X_out = step.transform(X_out)

        last = self.steps[-1][1]
        if y is not None and callable(getattr(last, "predict", None)):
            last.partial_fit(X_out, y)
        else:
            last.partial_fit(X_out)

        self.is_fitted_ = True
        return self

    def transform(self, X):
        """Transform input through all fitted steps without refitting.

        Parameters
        ----------
        X : array_like, shape (n_samples, n_features)
            Features accepted by the first step. Each later step receives the
            preceding transformer's output.

        Returns
        -------
        array_like, usually shape (n_samples, n_output_features)
            Output from the final transform method; exact type and shape follow
            the components.

        Raises
        ------
        ValueError
            If neither fit nor a nonempty partial_fit has completed, or the
            last step lacks transform.
        Exception
            Validation and computation errors raised by individual steps propagate.

        Notes
        -----
        Time: O(s) dispatch plus the sum of transform costs. Working space
        depends on the component outputs; intermediate results are not cached.
        """
        if not self.is_fitted_:
            raise ValueError("Call fit or partial_fit before transform")
        if not callable(getattr(self.steps[-1][1], "transform", None)):
            raise ValueError("The last step does not provide transform; use predict")

        X_out = X
        for name, step in self.steps:
            X_out = step.transform(X_out)
        return X_out

    def fit_transform(self, X, y=None):
        """Fit the pipeline and transform the same input.

        Parameters
        ----------
        X : array_like, shape (n_samples, n_features)
            Features accepted by the first step. Each later step receives the
            preceding transformer's output.
        y : array_like, usually shape (n_samples,), or None, default None
            Targets passed unchanged to the final step only if y is supplied
            and that step provides predict. Preprocessors receive X only.

        Returns
        -------
        array_like, usually shape (n_samples, n_output_features)
            Transformed training data returned by the final transformer.

        Raises
        ------
        ValueError
            If the final step lacks transform.
        Exception
            Validation and computation errors raised by individual steps propagate.

        Notes
        -----
        Time and space are those of fit followed by transform. This method
        reapplies intermediate transforms during the transform phase.
        Equivalent to self.fit(X, y).transform(X); use fit and predict when the
        last step is an estimator without transform.
        """
        if not callable(getattr(self.steps[-1][1], "transform", None)):
            raise ValueError("fit_transform requires a final transformer")
        self.fit(X, y)
        return self.transform(X)

    def predict(self, X):
        """Transform features and delegate prediction to the final estimator.

        Parameters
        ----------
        X : array_like, shape (n_samples, n_features)
            Features accepted by the first step. Each later step receives the
            preceding transformer's output.

        Returns
        -------
        array_like, usually shape (n_samples,)
            Predictions from the last step; exact type and shape depend on it.

        Raises
        ------
        ValueError
            If neither fit nor a nonempty partial_fit has completed, or the
            final step lacks predict. A model may also reject prediction while
            waiting for its first buffered training event.
        Exception
            Validation and computation errors raised by individual steps propagate.

        Notes
        -----
        Time: O(s) dispatch plus intermediate transform costs and the final
        predict cost. Space depends on intermediate arrays and estimator output.
        Uses the fitted preprocessing parameters; no step is fitted again.
        """
        if not self.is_fitted_:
            raise ValueError("Call fit or partial_fit before predict")
        last = self.steps[-1][1]
        if not callable(getattr(last, "predict", None)):
            raise ValueError("The last step does not provide predict; use transform")

        X_out = X
        for name, step in self.steps[:-1]:
            X_out = step.transform(X_out)
        return last.predict(X_out)

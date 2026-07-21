# Interactive application architecture

The contest interface is a thin Dash layer over package services. `create_app()` allocates an
independent session store and single-worker job registry. Browser state contains opaque session and
job IDs—not archive text, model output, API keys, or paths. Upload callbacks write a bounded ZIP to
a mode-0700 local session. Metadata preflight runs before the compile action becomes available.

The framework-neutral app service performs ZIP validation, graph normalization, optional
Professional-safe classification, chronological compilation, semantic analysis, rendering, and
manifest creation. Filtering is applied to Archive IR **before** either compiler sees it. The
original ZIP remains unchanged. A process-local thread worker prevents the long operation from
blocking the originating Dash callback; stage polling intentionally reports coarse stages rather
than fictional percentages.

Completed sessions contain scoped Archive IR, chronological HTML/PDF, semantic tables and atlas,
safety decisions, and an app manifest. The download bundle includes only these filtered products
and SHA-256 metadata. Advanced production concerns—multi-process queues, authentication, retention
automation, and remote object storage—are deliberately outside the local single-user design.

See [Privacy model](PRIVACY_MODEL.md), [deployment](DEPLOYMENT.md), and
[semantic evaluation](SEMANTIC_EVALUATION.md).

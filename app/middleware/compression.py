from starlette.middleware.gzip import GZipMiddleware


def setup_compression(app):
    """Add response compression"""
    app.add_middleware(GZipMiddleware, minimum_size=1000)
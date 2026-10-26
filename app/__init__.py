from __future__ import annotations


def create_app(test_config: dict | None = None) -> Flask:
    from flask import Flask, render_template

    from app.config import load_config

    app = Flask(__name__)
    app.config.update(load_config())
    if test_config:
        app.config.update(test_config)

    secret_key = app.config.get("SECRET_KEY")
    if (
        not isinstance(secret_key, str)
        or len(secret_key) < 32
        or secret_key.startswith("replace-")
    ):
        raise RuntimeError(
            "Set FLASK_SECRET_KEY to a random value at least 32 characters long."
        )

    from app.auth import auth_blueprint
    from app.routes import main_blueprint

    app.register_blueprint(auth_blueprint)
    app.register_blueprint(main_blueprint)

    @app.errorhandler(413)
    def request_too_large(error):
        del error
        return (
            render_template(
                "error.html",
                title="Upload too large",
                message=(
                    "The request exceeds the 5 MB upload limit. Choose a smaller "
                    "CSV file."
                ),
            ),
            413,
        )

    @app.errorhandler(400)
    def invalid_request(error):
        del error
        return (
            render_template(
                "error.html",
                title="Request not accepted",
                message="The form expired or the request was invalid. Reload the page and try again.",
            ),
            400,
        )

    return app

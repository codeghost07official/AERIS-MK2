import secrets
from functools import wraps
from typing import Any, Callable, TypeVar

from flask import (
    Blueprint,
    abort,
    current_app,
    redirect,
    render_template,
    request,
    session,
    url_for,
)
from werkzeug.security import check_password_hash


auth_blueprint = Blueprint("auth", __name__)
ViewFunction = TypeVar("ViewFunction", bound=Callable[..., Any])


def login_required(view: ViewFunction) -> ViewFunction:
    @wraps(view)
    def wrapped_view(*args: Any, **kwargs: Any) -> Any:
        if "username" not in session:
            return redirect(url_for("auth.login"))
        return view(*args, **kwargs)

    return wrapped_view  # type: ignore[return-value]


def valid_csrf_token() -> bool:
    submitted = request.form.get("csrf_token", "")
    expected = session.get("csrf_token", "")
    return bool(submitted and expected and secrets.compare_digest(submitted, expected))


@auth_blueprint.app_context_processor
def inject_csrf_token() -> dict[str, Callable[[], str]]:
    def csrf_token() -> str:
        token = session.get("csrf_token")
        if not token:
            token = secrets.token_urlsafe(32)
            session["csrf_token"] = token
        return token

    return {"csrf_token": csrf_token}


@auth_blueprint.route("/login", methods=["GET", "POST"])
def login() -> str | Any:
    error = None
    if request.method == "POST":
        if not valid_csrf_token():
            abort(400, description="The form expired. Reload the page and try again.")

        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        stored_hash = current_app.config["AERIS_USERS"].get(username)
        authenticated = False
        if stored_hash:
            try:
                authenticated = check_password_hash(stored_hash, password)
            except (TypeError, ValueError):
                current_app.logger.error(
                    "Invalid password hash configured for user %r.", username
                )

        if authenticated:
            session.clear()
            session["username"] = username
            return redirect(url_for("main.dashboard"))

        error = "The username or password is incorrect."

    return render_template(
        "login.html",
        error=error,
        login_configured=bool(current_app.config["AERIS_USERS"]),
    )


@auth_blueprint.post("/logout")
@login_required
def logout() -> Any:
    if not valid_csrf_token():
        abort(400, description="The form expired. Reload the page and try again.")
    session.clear()
    return redirect(url_for("auth.login"))

"""
ASGI config for modelImport project.

It exposes the ASGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/4.1/howto/deployment/asgi/
"""

import os
from daphne import get_default_application
from django.core.asgi import get_asgi_application

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "modelImport.settings")

django_application = get_asgi_application()
daphne_application = get_default_application()

application = daphne_application
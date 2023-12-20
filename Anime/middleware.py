from django.shortcuts import redirect

class RedirectIfAuthenticatedMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        # Check if the user is authenticated
        if request.user.is_authenticated:
            # Redirect to the home page if the user is already authenticated
            if request.path in ['/login/', '/register/']:
                return redirect('home')  # Adjust 'home' to your actual home URL or view name

        response = self.get_response(request)
        return response

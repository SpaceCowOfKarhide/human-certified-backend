self.addEventListener('install', e => {
    console.log('Service Worker: Installed');
});

self.addEventListener('activate', e => {
    console.log('Service Worker: Activated');
});

self.addEventListener('fetch', e => {
    // Optional: add caching here
    // e.respondWith(fetch(e.request));
});

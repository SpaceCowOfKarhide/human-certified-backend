const cacheName = 'ai-detector-cache-v1';
const assetsToCache = [
    './',
    './index.html',
    './manifest.json',
];

self.addEventListener('install', e => {
    e.waitUntil(
        caches.open(cacheName).then(cache => cache.addAll(assetsToCache))
    );
});

self.addEventListener('fetch', e => {
    e.respondWith(
        caches.match(e.request).then(response => response || fetch(e.request))
    );
});

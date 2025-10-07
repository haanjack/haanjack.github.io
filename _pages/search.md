---
layout: page
permalink: /search/
title: search
nav: true
nav_order: 7
---

<!-- Html Elements for Search -->
<div id="search-container">
    <input type="text" id="search-input"  class="form-control" placeholder="search...">
</div>

<br>

<div id="results-container">
    <ul></ul>
</div>

<!-- Script pointing to search.js -->
<!-- https://github.com/christian-fei/Simple-Jekyll-Search -->
<script src="/assets/js/search.js" type="text/javascript"></script>

<!-- Configuration -->
<script>
SimpleJekyllSearch({
    searchInput: document.getElementById('search-input'),
    resultsContainer: document.getElementById('results-container'),
    json: '/search.json'
})
</script>

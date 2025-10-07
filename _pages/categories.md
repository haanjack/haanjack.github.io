---
layout: page
permalink: /archive-categories/
title: Categories
description: Materials for courses you taught. Replace this text with your description.
---

<hr>

<div id="archive">
  {% for category in site.categories %}
  <div class="archive-group">
    <h2>{{ category | first }}</h2>
    {% for posts in category %}
      {% for post in posts %}
        {% if post.first[0] == null %}
        {% else %}
          <div class="archive-item">
            <div class="date">{{ post.date | date: '%b %d'}}</div>
            <a href="{{ post.url | prepend: site.baseurl }}">{{ post.title }}</a>
          </div>
        {% endif %}
      {% endfor %}
    {% endfor %}
  </div>
  {% endfor %}
</div>
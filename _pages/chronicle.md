---
layout: page
permalink: /archive-chronicle/
title: Chronicle
description: Materials for courses you taught. Replace this text with your description.
---

<hr>

<div id="archive">
	{% assign postsByYear = site.posts | group_by_exp:"post", "post.date | date: '%Y'"  %}
	{% for year in postsByYear %}
  <div class="archive-group">
	  <h3>{{ year.name }}</h3>
      <table class="table table-sm table-borderless">
      {% for post in year.items %}
      <tr>
        <th scope="row" width="90">{{ post.date | date: '%b %d'}}</th>
        <td>
          {% if item.inline -%}
            {{ item.content | remove: '<p>' | remove: '</p>' | emojify }}
          {%- else -%}
            <a class="news-title" href="{{ post.url | relative_url }}">{{ post.title }}</a>
          {%- endif %}
        </td>
      </tr>
      {% endfor %}
      </table>
  </div>
	{% endfor %}
</div>
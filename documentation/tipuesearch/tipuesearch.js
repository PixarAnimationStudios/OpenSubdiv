
/*
Tipue Search 3.0
Copyright (c) 2013 Tipue
Tipue Search is released under the MIT License
http://www.tipue.com/search
*/ 


(function($) {

     $.fn.tipuesearch = function(options) {

          var set = $.extend( {
          
               'show'                   : 7,
               'newWindow'              : false,
               'showURL'                : true,
               'minimumLength'          : 3,
               'descriptiveWords'       : 25,
               'highlightTerms'         : true,
               'highlightEveryTerm'     : false,
               'mode'                   : 'static',
               'liveDescription'        : '*',
               'liveContent'            : '*',
               'contentLocation'        : 'tipuesearch/tipuesearch_content.json'
          
          }, options);
          
          return this.each(function() {

               var tipuesearch_in = {
                    pages: []
               };
               $.ajaxSetup({
                    async: false
               });

               if (set.mode == 'live')
               {
                    for (var i = 0; i < tipuesearch_pages.length; i++)
                    {
                         $.get(tipuesearch_pages[i], '',
                              function (html)
                              {
                                   var cont = $(set.liveContent, html).text();
                                   cont = cont.replace(/\s+/g, ' ');
                                   var desc = $(set.liveDescription, html).text();
                                   desc = desc.replace(/\s+/g, ' ');
                                                                      
                                   var t_1 = html.toLowerCase().indexOf('<title>');
                                   var t_2 = html.toLowerCase().indexOf('</title>', t_1 + 7);
                                   if (t_1 != -1 && t_2 != -1)
                                   {
                                        var tit = html.slice(t_1 + 7, t_2);
                                   }
                                   else
                                   {
                                        var tit = 'No title';
                                   }

                                   tipuesearch_in.pages.push({
                                        "title": tit,
                                        "text": desc,
                                        "tags": cont,
                                        "loc": tipuesearch_pages[i] 
                                   });    
                              }
                         );
                    }
               }
               
               if (set.mode == 'json')
               {
                    $.getJSON(set.contentLocation,
                         function(json)
                         {
                              tipuesearch_in = $.extend({}, json);
                         }
                    );
               }

               if (set.mode == 'static')
               {
                    tipuesearch_in = $.extend({}, tipuesearch);
               }                              
               
               var tipue_search_w = '';
               if (set.newWindow)
               {
                    tipue_search_w = ' target="_blank"';      
               }

               function getURLP(name)
               {
                    return decodeURIComponent((new RegExp('[?|&]' + name + '=' + '([^&;]+?)(&|#|;|$)').exec(location.search)||[,""])[1].replace(/\+/g, '%20')) || null;
               }
               if (getURLP('q'))
               {
                    $('#tipue_search_input').val(getURLP('q'));
                    getTipueSearch(0, true);
               }               
                             
               $('#tipue_search_button').click(function()
               {
                    getTipueSearch(0, true);
               });
               $(this).keyup(function(event)
               {
                    if(event.keyCode == '13')
                    {
                         getTipueSearch(0, true);
                    }
               });

               function getTipueSearch(start, replace)
               {
                    $('#tipue_search_content').hide();
                    var out = '';
                    var results = '';
                    var show_replace = false;
                    var show_stop = false;
                    
                    var d = $('#tipue_search_input').val().toLowerCase();
                    d = $.trim(d);
                    var d_w = d.split(' ');
                    d = '';
                    for (var i = 0; i < d_w.length; i++)
                    {
                         var a_w = true;
                         for (var f = 0; f < tipuesearch_stop_words.length; f++)
                         {
                              if (d_w[i] == tipuesearch_stop_words[f])
                              {
                                   a_w = false;
                                   show_stop = true;          
                              }
                         }
                         if (a_w)
                         {
                              d = d + ' ' + d_w[i];
                         }
                    }
                    d = $.trim(d);
                    d_w = d.split(' ');
                    
                    if (d.length >= set.minimumLength)
                    {
                         if (replace)
                         {
                              var d_r = d;
                              for (var i = 0; i < d_w.length; i++)
                              {
                                   for (var f = 0; f < tipuesearch_replace.words.length; f++)
                                   {
                                        if (d_w[i] == tipuesearch_replace.words[f].word)
                                        {
                                             d = d.replace(d_w[i], tipuesearch_replace.words[f].replace_with);
                                             show_replace = true;
                                        }
                                   }
                              }
                              d_w = d.split(' ');
                         }                   
                         
                         var d_t = d;
                         for (var i = 0; i < d_w.length; i++)
                         {
                              for (var f = 0; f < tipuesearch_stem.words.length; f++)
                              {
                                   if (d_w[i] == tipuesearch_stem.words[f].word)
                                   {
                                        d_t = d_t + ' ' + tipuesearch_stem.words[f].stem;
                                   }
                              }
                         }
                         d_w = d_t.split(' ');

                         var c = 0;
                         found = new Array();
                         for (var i = 0; i < tipuesearch_in.pages.length; i++)
                         {
                              var score = 1000000000;
                              var s_t = tipuesearch_in.pages[i].text;
                              for (var f = 0; f < d_w.length; f++)
                              {
                                   var pat = new RegExp(d_w[f], 'i');
                                   if (tipuesearch_in.pages[i].title.search(pat) != -1)
                                   {
                                        score -= (200000 - i);
                                   }
                                   if (tipuesearch_in.pages[i].text.search(pat) != -1)
                                   {
                                        score -= (150000 - i);
                                   }
                                   
                                   if (set.highlightTerms)
                                   {
                                        if (set.highlightEveryTerm) 
                                        {
                                             var patr = new RegExp('(' + d_w[f] + ')', 'gi');
                                        }
                                        else
                                        {
                                             var patr = new RegExp('(' + d_w[f] + ')', 'i');
                                        }
                                        s_t = s_t.replace(patr, "<b>$1</b>");
                                   }
                                   if (tipuesearch_in.pages[i].tags.search(pat) != -1)
                                   {
                                        score -= (100000 - i);
                                   }
                    
                              }
                              if (score < 1000000000)
                              {
                                   found[c++] = score + '^' + tipuesearch_in.pages[i].title + '^' + s_t + '^' + tipuesearch_in.pages[i].loc;                                                                   
                              }
                         }                         
                         
                         if (c != 0)
                         {
                              if (show_replace == 1)
                              {
                                   out += '<div id="tipue_search_warning_head">Showing results for ' + d + '</div>';
                                   out += '<div id="tipue_search_warning">Search for <a href="javascript:void(0)" id="tipue_search_replaced">' + d_r + '</a></div>'; 
                              }
                              if (c == 1)
                              {
                                   out += '<div id="tipue_search_results_count">1 result</div>';
                              }
                              else
                              {
                                   c_c = c.toString().replace(/\B(?=(\d{3})+(?!\d))/g, ",");
                                   out += '<div id="tipue_search_results_count">' + c_c + ' results</div>';
                              }
                              
                              found.sort();
                              var l_o = 0;
                              for (var i = 0; i < found.length; i++)
                              {
                                   var fo = found[i].split('^');
                                   if (l_o >= start && l_o < set.show + start)
                                   {
                                        out += '<div class="tipue_search_content_title"><a href="' + fo[3] + '"' + tipue_search_w + '>' +  fo[1] + '</a></div>';
                                                                                
                                        var t = fo[2];
                                        var t_d = '';
                                        var t_w = t.split(' ');
                                        if (t_w.length < set.descriptiveWords)
                                        {
                                             t_d = t;
                                        }
                                        else
                                        {
                                             for (var f = 0; f < set.descriptiveWords; f++)
                                             {
                                                  t_d += t_w[f] + ' '; 	
                                             }
                                        }
                                        t_d = $.trim(t_d);
                                        if (t_d.charAt(t_d.length - 1) != '.')
                                        {
                                             t_d += ' ...';
                                        }
                                        out += '<div class="tipue_search_content_text">' + t_d + '</div>';
                                        
                                        if (set.showURL)
                                        {
                                             t_url = fo[3];
                                             if (t_url.length > 45)
                                             {
                                                  t_url = fo[3].substr(0, 45) + ' ...';
                                             }    
                                             out += '<div class="tipue_search_content_loc"><a href="' + fo[3] + '"' + tipue_search_w + '>' + t_url + '</a></div>';
                                        }
                                   }
                                   l_o++;     
                              }
                              
                              if (c > set.show)
                              {
                                   var pages = Math.ceil(c / set.show);
                                   var page = (start / set.show);
                                   out += '<div id="tipue_search_foot"><ul id="tipue_search_foot_boxes">';
                                   
                                   if (start > 0)
                                   {
                                       out += '<li><a href="javascript:void(0)" class="tipue_search_foot_box" id="' + (start - set.show) + '_' + replace + '">Prev</a></li>'; 
                                   }
                                                       
                                   if (page <= 2)
                                   {
                                        var p_b = pages;
                                        if (pages > 3)
                                        {
                                             p_b = 3;
                                        }                    
                                        for (var f = 0; f < p_b; f++)
                                        {
                                             if (f == page)
                                             {
                                                  out += '<li class="current">' + (f + 1) + '</li>';
                                             }
                                             else
                                             {
                                                  out += '<li><a href="javascript:void(0)" class="tipue_search_foot_box" id="' + (f * set.show) + '_' + replace + '">' + (f + 1) + '</a></li>';
                                             }
                                        }
                                   }
                                   else
                                   {
                                        var p_b = pages + 2;
                                        if (p_b > pages)
                                        {
                                             p_b = pages; 
                                        }
                                        for (var f = page; f < p_b; f++)
                                        {
                                             if (f == page)
                                             {
                                                  out += '<li class="current">' + (f + 1) + '</li>';
                                             }
                                             else
                                             {
                                                  out += '<li><a href="javascript:void(0)" class="tipue_search_foot_box" id="' + (f * set.show) + '_' + replace + '">' + (f + 1) + '</a></li>';
                                             }
                                        }                         
                                   }
                                                      
                                   if (page + 1 != pages)
                                   {
                                       out += '<li><a href="javascript:void(0)" class="tipue_search_foot_box" id="' + (start + set.show) + '_' + replace + '">Next</a></li>'; 
                                   }                    
                                   
                                   out += '</ul></div>';
                              }                        
                         }
                         else
                         {
                              out += '<div id="tipue_search_warning_head">Nothing found</div>'; 
                         }
                    }
                    else
                    {
                         if (show_stop)
                         {
                              out += '<div id="tipue_search_warning_head">Nothing found</div><div id="tipue_search_warning">Common words are largely ignored</div>';     
                         }
                         else
                         {
                              out += '<div id="tipue_search_warning_head">Search too short</div>';
                              if (set.minimumLength == 1)
                              {
                                   out += '<div id="tipue_search_warning">Should be one character or more</div>';
                              }
                              else
                              {
                                   out += '<div id="tipue_search_warning">Should be ' + set.minimumLength + ' characters or more</div>';
                              }
                         }
                    }
               
                    $('#tipue_search_content').html(out);
                    $('#tipue_search_content').slideDown(200);
                    
                    $('#tipue_search_replaced').click(function()
                    {
                         getTipueSearch(0, false);
                    });                
               
                    $('.tipue_search_foot_box').click(function()
                    {
                         var id_v = $(this).attr('id');
                         var id_a = id_v.split('_');
                    
                         getTipueSearch(parseInt(id_a[0]), id_a[1]);
                    });                                                       
               }          
          
          });
     };
   
})(jQuery);





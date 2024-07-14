using System;
using System.Text.RegularExpressions;
using Avalonia.Controls;
using Avalonia.Controls.Documents;
using Avalonia.Interactivity;
using Avalonia.Media;

namespace AvaloniaApplication2;

public partial class MainWindow : Window
{
    private string pdfPath;
    public async void OnPickPdfButtonClicked(object? sender, RoutedEventArgs routedEventArgs)
    {
        pdfPath = await PdfLoader.PickPdfFileAsync(this);
        if (pdfPath != null)
        {
            Console.WriteLine($"Selected PDF: {pdfPath}");
            PdfContentTextBlock.Text = PdfLoader.toText(pdfPath);
        }
        else
        {
            Console.WriteLine("No PDF selected");
        }
    }
    
    public MainWindow()
    {
        InitializeComponent();
    }
    
    private void SearchButton_Click(object sender, RoutedEventArgs e)
    {
        string searchText = SearchTextBox.Text;
        string content = PdfContentTextBlock.Text;

        PdfContentTextBlock.Text = "";
        PdfContentTextBlock.Inlines = new InlineCollection();
        
        var regex = new Regex(Regex.Escape(searchText), RegexOptions.IgnoreCase);
        var matches = regex.Matches(content);

        int lastIndex = 0;
        MatchCountTextBlock.Text = $"Matches Found: {matches.Count}";

        foreach (Match match in matches)
        {
            // Add the text before the match
            if (match.Index > lastIndex)
            {
                PdfContentTextBlock.Inlines.Add(new Run(content.Substring(lastIndex, match.Index - lastIndex)));
            }
            // Add the highlighted match
            if (match.Index == lastIndex)
            {
                Background = new SolidColorBrush(Colors.Yellow);
            };
            PdfContentTextBlock.Inlines.Add(new Run(match.Value){
                Background = new SolidColorBrush(Colors.Yellow)
            });

            lastIndex = match.Index + match.Length;
        }

        if (matches.Count == 0)
        {
            PdfContentTextBlock.Text = PdfLoader.toText(pdfPath);
        }
        if (lastIndex < content.Length)
        {
            PdfContentTextBlock.Inlines.Add(new Run(content.Substring(lastIndex)));
        }
        
    }
}
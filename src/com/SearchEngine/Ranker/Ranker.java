package com.SearchEngine.Ranker;

import java.io.FileReader;
import java.io.BufferedReader;
import java.util.ArrayList;
import java.util.HashMap;

public class Ranker {

    /* The number of pages in the graph*/
    private Integer pagesCount;

    /* The Graph adjacency list */
    private HashMap<Integer, ArrayList<Integer>> inList = new HashMap<Integer, ArrayList<Integer>>();

    /* The out degrees of each page */
    private ArrayList<Integer> outDegrees;

    /* The ranks of the pages */
    private ArrayList<Double> pagesRank;

    /* The dumping factor */
    private final Double alpha = 0.85;

    /* The maximum number iterations */
    private final Integer maxIterations = 100;

    /* Add an arc to the graph */
    private void addArc(int from, int to) {
        if (inList.containsKey(to))
            inList.get(to).add(from);
        outDegrees.set(from, outDegrees.get(from) + 1);
    }

    /* Initialize all vectors */
    private void initializeLists(int pagesCount) {
        this.pagesCount = pagesCount;

        outDegrees = new ArrayList<Integer>();
        pagesRank = new ArrayList<Double>();

        for (int i = 0; i < pagesCount; i++) {

            // Create a new list for each page
            ArrayList<Integer> newList = new ArrayList<Integer>();
            inList.put(i, newList);

            outDegrees.add(0);
            pagesRank.add(1.0 / pagesCount); // Initialize at first with 1/n prob
        }
    }

    private void readFile(String filePath) {
        try {
            BufferedReader br = new BufferedReader(new FileReader(filePath));
            String strLine;

            initializeLists(50);

            // Read file line by line
            while ((strLine = br.readLine()) != null) {

                String[] strs = strLine.trim().split(" ");
                Integer u = Integer.parseInt(strs[0]);
                Integer v = Integer.parseInt(strs[1]);

                // Add arcs
                this.addArc(u, v);
            }
        } catch (Exception e) {
            //Catch exception if any
            System.err.println("Error: " + e.getMessage());
        }
    }

    private void rankPages() {
        for (int iteration = 0; iteration < maxIterations; iteration++) {
            Double danglingSum, pagesRankSum = 0.0;

            for (int page = 0; page < pagesCount; page++) {
                pagesRankSum += pagesRank.get(page);
            }

            // Normalize the PR(i) needed for the power method calculations
            for (int page = 0; page < pagesCount; page++) {
                Double rank = pagesRank.get(page);
                pagesRank.set(page, rank * 1.0 / pagesRankSum);
            }

            pagesRankSum = 0.0;
            danglingSum = 0.0;

            for (int page = 0; page < pagesCount; page++) {
                if (outDegrees.get(page) == 0)
                    danglingSum += pagesRank.get(page);
                pagesRankSum += pagesRank.get(page);
            }

            System.out.println("PageRankSum " + pagesRankSum.toString() + ", DanglingSum " + danglingSum.toString());

            Double aPage = alpha * danglingSum * (1.0 / pagesCount); // Same for all pages
            Double oneProb = (1.0 - alpha) * (1.0 / pagesCount) * 1; // Same for all pages

            // Loop over all pages
            for (int page = 0; page < pagesCount; page++) {

                Double hPage = 0.0;

                if (inList.containsKey(page)) {
                    for (Integer from : inList.get(page)) {
                        hPage += (1.0 * pagesRank.get(from) / (1.0 * outDegrees.get(from)));
                    }
                    hPage *= alpha; // Multiply by dumping factor.
                }

                pagesRank.set(page, (hPage + aPage + oneProb));
            }
        }
    }

    // Debugging @Samir55
    private void debugging() {
        System.out.println("The saved edges printing");
        System.out.println("===========================================");
        for (Integer vertex : inList.keySet()) {
            System.out.println("For the vertex " + vertex.toString());
            System.out.println("OutDegrees " + outDegrees.get(vertex).toString());
            System.out.println("Vertices pointing to this vertex ");
            if (inList.get(vertex).isEmpty()) {
                System.out.println("nil");
            } else {
                for (Integer to : inList.get(vertex)) {
                    System.out.print(to.toString() + " ");
                }
                System.out.println();
            }
            System.out.println("===========================================");
        }
    }

    private void printPR() {
        Double checkSum = 0.0;
        for (Integer page = 0; page < pagesCount; page++) {
            checkSum += pagesRank.get(page);
            System.out.println(page.toString() + " = " + pagesRank.get(page));
        }
        System.out.println("checkSum = " + checkSum.toString());
    }

    private void savePR() {

    }

    public void run(String filePath) {
        readFile(filePath);
        rankPages();
        printPR();
    }
}

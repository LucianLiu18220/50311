/**
 *Submitted for verification at Etherscan.io on 2017-04-22
*/

pragma solidity ^0.4.10;

contract EtherGame 
{
    uint[] a;
    function Test1(uint a) public returns(address)
    {
        return msg.sender;
    }
    function Test2(uint a) returns(address)
    {
        return msg.sender;
    }
    function Test3(uint b) public returns(uint)
    {
        return a.length;
    }
    function Test4(uint b) returns(uint)
    {
        return a.length;
    }
    function Kill(uint a)
    {
        selfdestruct(msg.sender);
    }
}